# Stop Policy On-Device Integration (Plan B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make openpilot actually decelerate/stop for traffic lights & stop signs on the comma 4 (eGPU), driven by the v1 stop policy — supervised, param-gated, shipping defaulting to shadow (log-only), with the safe control path reusing openpilot's existing stop machinery.

**Architecture:** A new `stopmodeld` daemon reads the road camera over VisionIPC, runs a tinygrad-compiled combined model (DINOv2 backbone + GRU head, single-step with explicit hidden state), and publishes a `stopPolicy` cereal message. `plannerd`/`longitudinal_planner.py` consume it and, when active, only ever *lower* the cruise target (decelerate-only) and force a stop via the existing `shouldStop`/MPC path — so deceleration is comfort-bounded by the planner's existing accel clips. Two params gate it: `StopPolicyEnabled` (daemon runs + planner reads) and `StopPolicyActive` (planner applies vs. shadow-logs). Both default false → exact stock behavior.

**Tech Stack:** tinygrad (on-device runtime, `tinygrad.nn.onnx.OnnxRunner` — mirrors `selfdrive/modeld/compile_modeld.py`), cereal/msgq, PyTorch+DINOv2 (offline export only, in the stoppolicy venv), pytest.

**Parent spec:** `docs/superpowers/specs/2026-06-12-stop-sign-traffic-light-e2e-design.md`
**Builds on:** the trained v1 model (`runs/v1/best.pt`, `runs/v1/norm.json`) and `tools/stoppolicy/model.py`.

**Two Python environments (do not mix):**
- **stoppolicy venv** `/root/openpilot/tools/stoppolicy/.venv` (py3.13, torch+DINOv2+onnx) — Task 2 only (combined ONNX export).
- **openpilot venv** `/root/openpilot/.venv` (py3.12, built cereal/msgq, tinygrad submodule) — Tasks 1, 3, 4, 5, 6. Run its pytest as `/root/openpilot/.venv/bin/pytest` from `/root/openpilot`. If a compiled-module import fails, rebuild with `PATH=/root/openpilot/.venv/bin:$PATH scons -j$(nproc) <target>`.

**Honest scope limits (state these in the final README):**
- The model is **sim-only-trained**; real-road behavior is unproven. Shadow mode exists precisely to watch it before granting control.
- This dev box has a **CUDA** GPU, not the comma's **usbgpu**, and no car. Tinygrad compile + numeric parity are validated here on CUDA as a *proxy*; the actual comma-4/usbgpu run and the supervised drive are validated by the user on-device.

---

### Task 0: Environment & feasibility preflight

**Files:** none modified (preflight only).

- [ ] **Step 1: Confirm tinygrad + OnnxRunner import in the openpilot venv**

```bash
cd /root/openpilot
.venv/bin/python -c "from tinygrad.nn.onnx import OnnxRunner; from tinygrad.tensor import Tensor; from tinygrad.device import Device; print('tinygrad ok', Device.DEFAULT)"
```
Expected: prints `tinygrad ok <DEVICE>`. If the import fails, the submodule isn't on the path — run `git submodule update --init tinygrad_repo` and retry. Record the `Device.DEFAULT` value (likely `CUDA`); Task 3 uses it.

- [ ] **Step 2: Confirm the stoppolicy venv can load DINOv2 + the trained head**

```bash
/root/openpilot/tools/stoppolicy/.venv/bin/python -c "
import torch
from tools.stoppolicy.model import StopPolicyHead
m = StopPolicyHead(); m.load_state_dict(torch.load('runs/v1/best.pt', map_location='cpu', weights_only=False)['policy']); m.eval()
b = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', skip_validation=True).eval()
print('head+backbone ok')
" 2>&1 | tail -2
```
Run from `/root/openpilot`. Expected: `head+backbone ok`.

- [ ] **Step 3: Record findings** — report `Device.DEFAULT`, both env checks, and whether `runs/v1/best.pt` exists. No commit.

---

### Task 1: `stopPolicy` cereal message

**Files:**
- Modify: `cereal/custom.capnp` (struct `CustomReserved1`)
- Modify: `cereal/log.capnp` (union field `customReserved1 @108`)
- Modify: `cereal/services.py`
- Create: `selfdrive/stopmodeld/__init__.py`, `selfdrive/stopmodeld/tests/__init__.py` (empty)
- Test: `selfdrive/stopmodeld/tests/test_schema.py`

- [ ] **Step 1: Write the failing test** — `selfdrive/stopmodeld/tests/test_schema.py`:

```python
from cereal import messaging
from cereal.services import SERVICE_LIST


def test_stop_policy_roundtrip():
  dat = messaging.new_message('stopPolicy', valid=True)
  dat.stopPolicy.shouldStop = True
  dat.stopPolicy.desiredSpeed = 3.5
  dat.stopPolicy.distToStop = 12.0
  dat.stopPolicy.stopProb = 0.91
  dat.stopPolicy.modelValid = True
  dat.stopPolicy.frameId = 7
  evt = messaging.log_from_bytes(dat.to_bytes())
  assert evt.stopPolicy.shouldStop is True
  assert abs(evt.stopPolicy.desiredSpeed - 3.5) < 1e-5
  assert abs(evt.stopPolicy.distToStop - 12.0) < 1e-5
  assert abs(evt.stopPolicy.stopProb - 0.91) < 1e-5
  assert evt.stopPolicy.modelValid is True
  assert evt.stopPolicy.frameId == 7


def test_stop_policy_service():
  svc = SERVICE_LIST['stopPolicy']
  assert svc.should_log
  assert svc.frequency == 20.
```

- [ ] **Step 2: Run, verify FAIL** — `/root/openpilot/.venv/bin/pytest selfdrive/stopmodeld/tests/test_schema.py -v` (create the empty `__init__.py` files first). Expect KeyError / no such field.

- [ ] **Step 3: In `cereal/custom.capnp`**, replace:

```capnp
struct CustomReserved1 @0xaedffd8f31e7b55d {
}
```
with (identifier `@0x...` MUST NOT change):
```capnp
struct StopPolicy @0xaedffd8f31e7b55d {
  shouldStop @0 :Bool;     # force a stop now
  desiredSpeed @1 :Float32; # m/s, policy speed target (planner only lowers cruise to this)
  distToStop @2 :Float32;   # meters to the stop line (-1 if not stopping)
  stopProb @3 :Float32;     # raw sigmoid(stop_logit)
  modelValid @4 :Bool;      # frames fresh + model ran ok
  frameId @5 :UInt32;       # road-camera frame id this decision is for
}
```

- [ ] **Step 4: In `cereal/log.capnp`**, replace:
```capnp
    customReserved1 @108 :Custom.CustomReserved1;
```
with (the `@108` MUST NOT change):
```capnp
    stopPolicy @108 :Custom.StopPolicy;
```

- [ ] **Step 5: In `cereal/services.py`**, directly below the `"telephotoCameraState": (True, 40., 40),` line add:
```python
  "stopPolicy": (True, 20., 20),  # stop policy decision, fork service (CustomReserved1)
```

- [ ] **Step 6: Run the test → 2 PASSED**, then `/root/openpilot/.venv/bin/pytest cereal/ -x -q` (no new failures).

- [ ] **Step 7: Commit**
```bash
git add cereal/custom.capnp cereal/log.capnp cereal/services.py selfdrive/stopmodeld/
git commit -m "cereal: add stopPolicy fork message (CustomReserved1)"
```

---

### Task 2: Combined device model export (DINOv2 + head → ONNX)

Runs in the **stoppolicy venv**. Produces one ONNX taking a preprocessed image + speed + hidden state, returning the stop decision + next hidden state — the single-step streaming form the daemon runs every frame.

**Files:**
- Create: `tools/stoppolicy/export_device.py`
- Test: `tools/stoppolicy/tests/test_export_device.py`

- [ ] **Step 1: Write the failing test** — `tools/stoppolicy/tests/test_export_device.py`:

```python
import numpy as np
import torch

from tools.stoppolicy.export_device import build_combined, export_combined
from tools.stoppolicy.model import HIDDEN_DIM

IMG_H, IMG_W = 252, 448


def test_combined_forward_shapes():
  m = build_combined(policy_state=None, device="cpu").eval()
  img = torch.rand(1, 3, IMG_H, IMG_W)
  speed = torch.tensor([[5.0]])
  h = torch.zeros(1, HIDDEN_DIM)
  stop_logit, dist, dspeed, h_out = m(img, speed, h)
  assert stop_logit.shape == (1, 1)
  assert dist.shape == (1, 1)
  assert dspeed.shape == (1, 1)
  assert h_out.shape == (1, HIDDEN_DIM)


def test_export_and_parity(tmp_path):
  m = build_combined(policy_state=None, device="cpu").eval()
  path = tmp_path / "stop_device.onnx"
  export_combined(m, path)

  import onnxruntime as ort
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  img = np.random.rand(1, 3, IMG_H, IMG_W).astype(np.float32)
  speed = np.array([[5.0]], dtype=np.float32)
  h = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
  outs = sess.run(None, {"image": img, "speed": speed, "hidden_in": h})
  with torch.no_grad():
    t = m(torch.from_numpy(img), torch.from_numpy(speed), torch.from_numpy(h))
  for o, tt in zip(outs, t):
    np.testing.assert_allclose(o, tt.numpy(), atol=1e-3)
```

- [ ] **Step 2: Run, verify FAIL** — `cd /root/openpilot && tools/stoppolicy/.venv/bin/python -m pytest tools/stoppolicy/tests/test_export_device.py -v`.

- [ ] **Step 3: Implement `tools/stoppolicy/export_device.py`:**

```python
"""Export the combined on-device model: image + speed + hidden -> stop decision.

The DINOv2 backbone (frozen) and the trained GRU head become one ONNX graph in
single-step streaming form, matching what stopmodeld runs each frame. Input image
is already preprocessed to (1,3,252,448) float32 in [0,1]; ImageNet normalization
happens inside the graph so the daemon only has to resize+scale.
"""
import torch
import torch.nn as nn

from tools.stoppolicy.model import StopPolicyHead, HIDDEN_DIM

IMG_H, IMG_W = 252, 448
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class CombinedStopModel(nn.Module):
  def __init__(self, backbone, head):
    super().__init__()
    self.backbone = backbone
    self.head = head
    self.register_buffer("mean", _MEAN)
    self.register_buffer("std", _STD)

  def forward(self, image, speed, hidden_in):
    x = (image - self.mean) / self.std
    feats = self.backbone.forward_features(x)
    cls_tok = feats["x_norm_clstoken"]          # (1, 384)
    patch_mean = feats["x_norm_patchtokens"].mean(dim=1)  # (1, 384)
    feat = torch.cat([cls_tok, patch_mean], dim=1)        # (1, 768)
    out, hidden_out = self.head.step(feat, speed, hidden_in)
    return out["stop_logit"], out["dist_to_stop"], out["desired_speed"], hidden_out


def build_combined(policy_state, device="cpu"):
  backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', skip_validation=True)
  head = StopPolicyHead()
  if policy_state is not None:
    head.load_state_dict(policy_state)
  m = CombinedStopModel(backbone, head).to(device).eval()
  for p in m.parameters():
    p.requires_grad_(False)
  return m


def export_combined(model, path):
  model.eval()
  img = torch.zeros(1, 3, IMG_H, IMG_W)
  speed = torch.zeros(1, 1)
  hidden = torch.zeros(1, HIDDEN_DIM)
  torch.onnx.export(
    model, (img, speed, hidden), str(path),
    input_names=["image", "speed", "hidden_in"],
    output_names=["stop_logit", "dist_to_stop", "desired_speed", "hidden_out"],
    opset_version=17, dynamo=False,
  )


def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--out", required=True)
  args = parser.parse_args()
  state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["policy"]
  m = build_combined(policy_state=state, device="cpu")
  export_combined(m, args.out)
  print(f"exported combined device model -> {args.out}")


if __name__ == "__main__":
  main()
```

If `forward_features` tracing under `torch.onnx.export` errors (DINOv2 control flow), the fix is to keep `dynamo=False` (already set) and, if needed, pass `do_constant_folding=True`; document any change. If a specific DINOv2 op refuses to export to opset 17, report it precisely — that same op is the Task 3 risk.

- [ ] **Step 4: Tests pass (2).** Print the ONNX file size in your report (DINOv2 ViT-S/14 ≈ 80–90 MB fp32).

- [ ] **Step 5: Commit** — `git add tools/stoppolicy/ && git commit -m "stoppolicy: combined on-device model export (DINOv2 + head)"`

---

### Task 3: Tinygrad compile of the combined model (FEASIBILITY GATE)

Runs in the **openpilot venv** on whatever `Device.DEFAULT` is (CUDA here; usbgpu on-device). Loads the combined ONNX with tinygrad's `OnnxRunner`, JITs it, verifies numeric parity vs onnxruntime, and pickles a `.pkl` the daemon will load. **This is the make-or-break step** — if tinygrad cannot run a DINOv2 op, on-device deployment with this backbone is blocked; report it precisely.

**Files:**
- Create: `tools/stoppolicy/compile_device.py`
- Test: `tools/stoppolicy/tests/test_compile_device.py`

- [ ] **Step 1: Implement `tools/stoppolicy/compile_device.py`:**

```python
"""Compile the combined stop model ONNX to a tinygrad JIT pickle for on-device use.

Mirrors selfdrive/modeld/compile_modeld.py at small scale: OnnxRunner -> TinyJit ->
capture/replay -> pickle. Runs on Device.DEFAULT (CUDA on a dev box, the usbgpu on
the comma device).
"""
import argparse
import pickle
import time

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.onnx import OnnxRunner

IMG_H, IMG_W, HIDDEN_DIM = 252, 448, 256


def _rand_inputs(seed):
  rng = np.random.default_rng(seed)
  return {
    "image": rng.random((1, 3, IMG_H, IMG_W), dtype=np.float32),
    "speed": rng.random((1, 1), dtype=np.float32),
    "hidden_in": np.zeros((1, HIDDEN_DIM), dtype=np.float32),
  }


def build_runner(onnx_path):
  runner = OnnxRunner(onnx_path)

  def run(image, speed, hidden_in):
    outs = runner({"image": image, "speed": speed, "hidden_in": hidden_in})
    return tuple(v.cast('float32') for v in outs.values())

  return TinyJit(run, prune=True)


def compile_device(onnx_path, out_path):
  jit = build_runner(onnx_path)
  # warm + capture (3 runs so TinyJit captures the graph)
  last = None
  for i in range(3):
    inp = {k: Tensor(v, device=Device.DEFAULT) for k, v in _rand_inputs(0).items()}
    Device.default.synchronize()
    st = time.perf_counter()
    outs = jit(**inp)
    vals = [o.numpy() for o in outs]
    Device.default.synchronize()
    print(f"  [{i+1}/3] {(time.perf_counter()-st)*1e3:.1f} ms")
    last = vals
  jit = pickle.loads(pickle.dumps(jit))  # pickle round-trip
  with open(out_path, "wb") as f:
    pickle.dump(jit, f)
  print(f"saved {out_path}")
  return last


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--onnx", required=True)
  p.add_argument("--out", required=True)
  args = p.parse_args()
  compile_device(args.onnx, args.out)


if __name__ == "__main__":
  main()
```

- [ ] **Step 2: Write `tools/stoppolicy/tests/test_compile_device.py`** (marked slow; needs a real ONNX → generates one from a randomly-initialized combined model in the stoppolicy venv is NOT available here, so the test takes an onnx path via env or builds a tiny stand-in). Use this approach — build the combined ONNX once via a session-scoped fixture that shells out to the stoppolicy venv:

```python
import os
import subprocess

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SP_PY = os.path.join(REPO, "tools/stoppolicy/.venv/bin/python")


@pytest.fixture(scope="module")
def combined_onnx(tmp_path_factory):
  out = tmp_path_factory.mktemp("dev") / "stop_device.onnx"
  # build an untrained combined model ONNX using the stoppolicy venv (has torch+DINOv2)
  code = (
    "from tools.stoppolicy.export_device import build_combined, export_combined;"
    f"export_combined(build_combined(policy_state=None, device='cpu'), r'{out}')"
  )
  subprocess.run([SP_PY, "-c", code], cwd=REPO, check=True)
  return str(out)


@pytest.mark.slow
def test_tinygrad_parity(combined_onnx, tmp_path):
  import onnxruntime as ort
  from tools.stoppolicy.compile_device import compile_device, _rand_inputs

  tg_vals = compile_device(combined_onnx, str(tmp_path / "m.pkl"))
  sess = ort.InferenceSession(combined_onnx, providers=["CPUExecutionProvider"])
  ort_vals = sess.run(None, _rand_inputs(0))
  # tinygrad ran on _rand_inputs(0) too (seed 0); compare stop_logit + hidden
  np.testing.assert_allclose(tg_vals[0], ort_vals[0], atol=2e-2)
  np.testing.assert_allclose(tg_vals[3], ort_vals[3], atol=2e-2)
```

(`onnxruntime` must be importable in the openpilot venv; if not, `/root/openpilot/.venv/bin/pip install --no-cache-dir onnxruntime`. Tolerance 2e-2 reflects fp32 GPU kernel differences across runtimes.)

- [ ] **Step 3: Run** `/root/openpilot/.venv/bin/pytest tools/stoppolicy/tests/test_compile_device.py -v -m slow`.
  - **If it PASSES:** DINOv2 runs under tinygrad on this backend. Record per-run latency from the printed ms (the ~10 Hz budget question). Continue.
  - **If it FAILS** on an unsupported tinygrad op: STOP, report **DONE_WITH_CONCERNS** with the exact op/error. Do not fake a pass. The control-path tasks (4–5) can still proceed against a stub model; flag that on-car deployment is blocked pending a tinygrad-compatible backbone.

- [ ] **Step 4: Commit** — `git add tools/stoppolicy/ && git commit -m "stoppolicy: tinygrad compile of combined device model"`

---

### Task 4: `stopmodeld` daemon

Runs the model each road-camera frame, manages hidden state, applies stop hysteresis, publishes `stopPolicy`. The model runner is **injectable** so the daemon's logic is fully testable here without a tinygrad `.pkl` or a camera.

**Files:**
- Create: `selfdrive/stopmodeld/stopmodeld.py`
- Test: `selfdrive/stopmodeld/tests/test_stopmodeld.py`

Design contract:
- `StopDecider(enter_thresh=0.6, exit_thresh=0.4)`: pure hysteresis on `stop_prob`; `update(stop_prob) -> bool shouldStop`. Stays stopped between exit and enter thresholds (no brake oscillation).
- `preprocess_nv12(buf_bytes, width, height) -> np.ndarray (1,3,252,448) float32 [0,1]`: NV12 → RGB → resize. Pure, testable.
- `StopModeld(runner, decider=None)`: `runner(image, speed, hidden) -> (stop_logit, dist, desired_speed, hidden_out)` (numpy). Holds hidden state (zeros init), exposes `step(image, speed) -> dict` building the stopPolicy fields (sigmoid, hysteresis, desiredSpeed clamped ≥0, distToStop), and `publish(pm, fields, frame_id, valid)`.
- `main()`: VisionIpcClient on `VISION_STREAM_ROAD`, SubMaster(['carState']) for speed, real runner loads the `.pkl` via tinygrad; gated on `Params().get_bool("StopPolicyEnabled")` (exit cleanly if off). Mirror `dmonitoringmodeld.py`.

- [ ] **Step 1: Write the failing test** — `selfdrive/stopmodeld/tests/test_stopmodeld.py`:

```python
import numpy as np

from openpilot.selfdrive.stopmodeld.stopmodeld import StopDecider, preprocess_nv12, StopModeld
from tools.stoppolicy.model import HIDDEN_DIM


def test_hysteresis():
  d = StopDecider(enter_thresh=0.6, exit_thresh=0.4)
  assert d.update(0.5) is False          # below enter
  assert d.update(0.7) is True           # crosses enter
  assert d.update(0.5) is True           # holds in the band
  assert d.update(0.3) is False          # drops below exit


def test_preprocess_shape():
  w, h = 320, 240
  nv12 = np.full((h * 3 // 2, w), 128, dtype=np.uint8).tobytes()
  out = preprocess_nv12(nv12, w, h)
  assert out.shape == (1, 3, 252, 448)
  assert out.dtype == np.float32
  assert 0.0 <= out.min() and out.max() <= 1.0


def test_stopmodeld_step_builds_fields_and_threads_hidden():
  seen = {}
  def runner(image, speed, hidden):
    seen["hidden_in"] = hidden.copy()
    # stop_logit large positive -> sigmoid ~1
    return (np.array([[5.0]], np.float32), np.array([[8.0]], np.float32),
            np.array([[2.0]], np.float32), (hidden + 1.0).astype(np.float32))
  m = StopModeld(runner=runner)
  img = np.zeros((1, 3, 252, 448), np.float32)
  f1 = m.step(img, speed=10.0)
  assert f1["shouldStop"] is True
  assert f1["stopProb"] > 0.99
  assert abs(f1["desiredSpeed"] - 2.0) < 1e-5
  assert abs(f1["distToStop"] - 8.0) < 1e-5
  assert np.allclose(seen["hidden_in"], 0.0)   # first call: zero hidden
  m.step(img, speed=10.0)
  assert np.allclose(seen["hidden_in"], 1.0)   # hidden threaded from prev step


def test_desired_speed_never_negative():
  def runner(image, speed, hidden):
    return (np.array([[-5.0]], np.float32), np.array([[-1.0]], np.float32),
            np.array([[-3.0]], np.float32), np.zeros((1, HIDDEN_DIM), np.float32))
  m = StopModeld(runner=runner)
  f = m.step(np.zeros((1, 3, 252, 448), np.float32), speed=0.0)
  assert f["desiredSpeed"] >= 0.0
  assert f["shouldStop"] is False  # sigmoid(-5) < enter
```

- [ ] **Step 2: Run, verify FAIL** — `/root/openpilot/.venv/bin/pytest selfdrive/stopmodeld/tests/test_stopmodeld.py -v`.

- [ ] **Step 3: Implement `selfdrive/stopmodeld/stopmodeld.py`.** Pure pieces exactly as the contract; `main()` mirrors `dmonitoringmodeld.py` (VisionIpcClient recv loop, SubMaster carState, PubMaster stopPolicy, param gate). The real runner loads the Task 3 `.pkl`:

```python
#!/usr/bin/env python3
import os
import math
import pickle

import numpy as np
import cv2

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

MODEL_W, MODEL_H, HIDDEN_DIM = 448, 252, 256


def _sigmoid(x):
  return 1.0 / (1.0 + math.exp(-x))


class StopDecider:
  def __init__(self, enter_thresh=0.6, exit_thresh=0.4):
    self.enter_thresh = enter_thresh
    self.exit_thresh = exit_thresh
    self.stopped = False

  def update(self, stop_prob):
    if self.stopped:
      if stop_prob < self.exit_thresh:
        self.stopped = False
    else:
      if stop_prob > self.enter_thresh:
        self.stopped = True
    return self.stopped


def preprocess_nv12(buf_bytes, width, height):
  yuv = np.frombuffer(buf_bytes, dtype=np.uint8)[:width * height * 3 // 2].reshape(height * 3 // 2, width)
  rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)
  rgb = cv2.resize(rgb, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
  return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


class StopModeld:
  def __init__(self, runner, decider=None):
    self.runner = runner
    self.decider = decider or StopDecider()
    self.hidden = np.zeros((1, HIDDEN_DIM), dtype=np.float32)

  def step(self, image, speed):
    stop_logit, dist, dspeed, hidden_out = self.runner(
      image, np.array([[speed]], np.float32), self.hidden)
    self.hidden = np.asarray(hidden_out, np.float32)
    prob = _sigmoid(float(stop_logit.flatten()[0]))
    return {
      "shouldStop": bool(self.decider.update(prob)),
      "stopProb": float(prob),
      "desiredSpeed": max(0.0, float(dspeed.flatten()[0])),
      "distToStop": float(dist.flatten()[0]),
    }

  def publish(self, pm, fields, frame_id, valid):
    msg = messaging.new_message("stopPolicy", valid=True)
    msg.stopPolicy.shouldStop = fields["shouldStop"] and valid
    msg.stopPolicy.desiredSpeed = fields["desiredSpeed"]
    msg.stopPolicy.distToStop = fields["distToStop"]
    msg.stopPolicy.stopProb = fields["stopProb"]
    msg.stopPolicy.modelValid = valid
    msg.stopPolicy.frameId = int(frame_id)
    pm.send("stopPolicy", msg)


def _tinygrad_runner(pkl_path):
  from tinygrad.tensor import Tensor
  from tinygrad.device import Device
  with open(pkl_path, "rb") as f:
    jit = pickle.load(f)

  def runner(image, speed, hidden):
    outs = jit(image=Tensor(image, device=Device.DEFAULT),
               speed=Tensor(speed, device=Device.DEFAULT),
               hidden_in=Tensor(hidden, device=Device.DEFAULT))
    return tuple(np.asarray(o.numpy(), np.float32) for o in outs)
  return runner


def main():
  if not Params().get_bool("StopPolicyEnabled"):
    cloudlog.warning("stopmodeld: StopPolicyEnabled not set, exiting")
    return

  from msgq.visionipc import VisionIpcClient, VisionStreamType
  pkl_path = os.path.join(os.path.dirname(__file__), "models", "stop_device.pkl")
  model = StopModeld(runner=_tinygrad_runner(pkl_path))

  vipc = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while not vipc.connect(False):
    pass
  sm = messaging.SubMaster(["carState"])
  pm = messaging.PubMaster(["stopPolicy"])

  while True:
    buf = vipc.recv()
    if buf is None:
      continue
    sm.update(0)
    image = preprocess_nv12(bytes(buf.data), buf.width, buf.height)
    fields = model.step(image, speed=sm["carState"].vEgo)
    model.publish(pm, fields, vipc.frame_id, valid=sm.all_alive(["carState"]))


if __name__ == "__main__":
  main()
```

(If `cv2` is missing in the openpilot venv, `/root/openpilot/.venv/bin/pip install --no-cache-dir opencv-python-headless`. If `VisionIpcClient.connect` signature differs, match `dmonitoringmodeld.py`.)

- [ ] **Step 4: Tests pass (4).** Commit — `git add selfdrive/stopmodeld/ && git commit -m "stopmodeld: stop-policy daemon (hysteresis, hidden-state, stopPolicy publish)"`

---

### Task 5: Longitudinal planner integration (param-gated, decelerate-only)

**Files:**
- Modify: `selfdrive/controls/plannerd.py` (subscribe `stopPolicy`)
- Modify: `selfdrive/controls/lib/longitudinal_planner.py` (apply, gated)
- Test: `selfdrive/controls/lib/tests/test_stop_policy_planner.py`

Safety rules baked into code:
- Reads `StopPolicyEnabled` / `StopPolicyActive` from Params at planner init and re-reads every ~1 s (counter), never per-cycle.
- Effect when **active + enabled + msg fresh + modelValid**: `v_cruise = min(v_cruise, max(0.0, desiredSpeed))` and, if `shouldStop`, `v_cruise = 0.0` (mirrors the existing `force_slow_decel` path at line 128-129). It can ONLY lower `v_cruise` — never raise it.
- **Shadow** (enabled, not active): compute the would-be `v_cruise` and log via `cloudlog`, but do NOT modify the real `v_cruise`.
- Stale `stopPolicy` (not updated/alive) or `modelValid==False` or feature disabled: strict no-op → stock behavior.

- [ ] **Step 1: Write the failing test** — `selfdrive/controls/lib/tests/test_stop_policy_planner.py`. Test the pure decision helper so we don't need a full MPC:

```python
from openpilot.selfdrive.controls.lib.longitudinal_planner import apply_stop_policy


def test_disabled_is_noop():
  v = apply_stop_policy(20.0, enabled=False, active=False, fresh=True,
                        model_valid=True, should_stop=True, desired_speed=0.0)
  assert v == 20.0


def test_shadow_does_not_change_v_cruise():
  v = apply_stop_policy(20.0, enabled=True, active=False, fresh=True,
                        model_valid=True, should_stop=True, desired_speed=0.0)
  assert v == 20.0


def test_active_should_stop_forces_zero():
  v = apply_stop_policy(20.0, enabled=True, active=True, fresh=True,
                        model_valid=True, should_stop=True, desired_speed=3.0)
  assert v == 0.0


def test_active_lowers_to_desired_speed():
  v = apply_stop_policy(20.0, enabled=True, active=True, fresh=True,
                        model_valid=True, should_stop=False, desired_speed=8.0)
  assert v == 8.0


def test_never_raises_v_cruise():
  v = apply_stop_policy(5.0, enabled=True, active=True, fresh=True,
                        model_valid=True, should_stop=False, desired_speed=18.0)
  assert v == 5.0  # desired higher than cruise -> unchanged


def test_stale_or_invalid_is_noop():
  assert apply_stop_policy(15.0, enabled=True, active=True, fresh=False,
                           model_valid=True, should_stop=True, desired_speed=0.0) == 15.0
  assert apply_stop_policy(15.0, enabled=True, active=True, fresh=True,
                           model_valid=False, should_stop=True, desired_speed=0.0) == 15.0
```

- [ ] **Step 2: Run, verify FAIL** — `/root/openpilot/.venv/bin/pytest selfdrive/controls/lib/tests/test_stop_policy_planner.py -v`.

- [ ] **Step 3: Add `apply_stop_policy` to `longitudinal_planner.py`** (module-level pure function, above the class):

```python
def apply_stop_policy(v_cruise, enabled, active, fresh, model_valid, should_stop, desired_speed):
  """Decelerate-only stop-policy effect on the cruise target.

  Returns a possibly-lowered v_cruise. Never raises it. No-op unless the feature
  is enabled+active with a fresh, valid message. Shadow mode (enabled, not active)
  is a no-op here; the caller logs the would-be value.
  """
  if not (enabled and active and fresh and model_valid):
    return v_cruise
  if should_stop:
    return 0.0
  return min(v_cruise, max(0.0, desired_speed))
```

- [ ] **Step 4: Wire it into `LongitudinalPlanner`.** In `__init__`, add param plumbing:

```python
    from openpilot.common.params import Params
    self._params = Params()
    self._sp_enabled = self._params.get_bool("StopPolicyEnabled")
    self._sp_active = self._params.get_bool("StopPolicyActive")
    self._sp_counter = 0
```

In `update(self, sm)`, immediately after the `if force_slow_decel: v_cruise = 0.0` block (line ~128-129), insert:

```python
    # stop policy (decelerate-only, param-gated; re-read params ~1 Hz)
    self._sp_counter += 1
    if self._sp_counter % 20 == 0:
      self._sp_enabled = self._params.get_bool("StopPolicyEnabled")
      self._sp_active = self._params.get_bool("StopPolicyActive")
    if self._sp_enabled and 'stopPolicy' in sm.data:
      sp = sm['stopPolicy']
      fresh = sm.alive['stopPolicy'] and sm.valid['stopPolicy']
      new_v = apply_stop_policy(v_cruise, self._sp_enabled, self._sp_active, fresh,
                                sp.modelValid, sp.shouldStop, sp.desiredSpeed)
      if self._sp_active:
        v_cruise = new_v
      elif fresh and sp.modelValid and new_v < v_cruise:
        cloudlog.info(f"stopPolicy SHADOW: would set v_cruise {v_cruise:.1f} -> {new_v:.1f} (stop={sp.shouldStop})")
```

- [ ] **Step 5: Subscribe in `plannerd.py`** — add `'stopPolicy'` to the SubMaster list (line 22-23). It must NOT be in the `poll` set and must be optional so a missing publisher never blocks the planner:

```python
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'liveParameters', 'radarState', 'modelV2', 'selfdriveState', 'stopPolicy'],
                           poll='modelV2', ignore_alive=['stopPolicy'], ignore_avg_freq=['stopPolicy'])
```
(Confirm `SubMaster` accepts `ignore_alive`/`ignore_avg_freq` — they're standard; match existing usage in the repo.)

- [ ] **Step 6: Tests pass (6).** Also run the existing planner tests to prove no regression: `/root/openpilot/.venv/bin/pytest selfdrive/controls/lib/tests/ -q` (and `selfdrive/test/process_replay` longitudinal test if quick).

- [ ] **Step 7: Commit** — `git add selfdrive/controls/ && git commit -m "longitudinal_planner: param-gated decelerate-only stop policy + shadow mode"`

---

### Task 6: Manager wiring, safety doc, deployment README

**Files:**
- Modify: `system/manager/process_config.py`
- Create: `selfdrive/stopmodeld/README.md`

- [ ] **Step 1: Add the daemon to the manager.** Below the `TELECAM` flag (or near other modeld entries) in `system/manager/process_config.py`:

```python
STOP_POLICY = os.getenv("STOP_POLICY") is not None or Params().get_bool("StopPolicyEnabled")
```
and in `procs`, below `dmonitoringmodeld`:
```python
  PythonProcess("stopmodeld", "selfdrive.stopmodeld.stopmodeld", only_onroad, enabled=STOP_POLICY),
```
(If `Params()` isn't already imported/usable at module load in process_config, gate on the env var alone — `os.getenv("STOP_POLICY")` — and document that on-device the launch wrapper sets it from the param. Verify against how other param-gated procs do it.)

- [ ] **Step 2: Verify config parses** — `/root/openpilot/.venv/bin/python -c "from openpilot.system.manager.process_config import procs; print([p.name for p in procs if p.name=='stopmodeld'])"` → `['stopmodeld']`.

- [ ] **Step 3: Write `selfdrive/stopmodeld/README.md`** covering: architecture (daemon → stopPolicy → planner), the two params, **shadow-first procedure**, the safety invariants (decelerate-only, comfort-bounded via existing MPC clips, driver brake/gas override, stale→fallback), the sim-only-model caveat, and the on-device build/deploy steps:

````markdown
# stopmodeld — on-device stop policy

Runs the trained stop policy on the road camera and asks the longitudinal planner
to stop for lights/signs. Supervised, param-gated, decelerate-only.

## Build the device model (on the comma 4 / with usbgpu)
```sh
# 1. export combined model (needs torch venv)
tools/stoppolicy/.venv/bin/python -m tools.stoppolicy.export_device \
    --checkpoint runs/v1/best.pt --out /tmp/stop_device.onnx
# 2. compile to tinygrad for the device GPU
.venv/bin/python -m tools.stoppolicy.compile_device \
    --onnx /tmp/stop_device.onnx --out selfdrive/stopmodeld/models/stop_device.pkl
```

## Enable (shadow first!)
```sh
# shadow: daemon runs, planner LOGS what it would do, no control authority
echo -n 1 > /data/params/d/StopPolicyEnabled
# after reviewing shadow logs on real roads, grant control (supervised):
echo -n 1 > /data/params/d/StopPolicyActive
```

## Safety invariants (enforced in code)
- Planner can only LOWER the cruise target — never raise it (decelerate-only).
- Deceleration uses openpilot's existing stop/MPC path → comfort-bounded by the
  planner's accel clips; no accel-limit bypass.
- Driver brake/gas overrides as always; disengage is unaffected.
- Stale/invalid `stopPolicy`, or either param off → exact stock behavior.

## Caveats
- The v1 model is **simulator-trained**; real-road performance is unproven. Run
  shadow mode and supervise every engaged drive.
````

- [ ] **Step 4: Commit** — `git add system/manager/process_config.py selfdrive/stopmodeld/README.md && git commit -m "stopmodeld: manager wiring + deployment/safety README"`

---

## Out of scope (later)
- Telephoto-camera fusion (this daemon uses the road cam; telephoto features are a deliberate later FEAT_DIM change once Plan A logs real data).
- Real-drive auto-labeling / retraining off shadow logs.
- Precise stop-line placement via the MPC (v1 brakes to a stop when shouldStop; distToStop is published for later use).
- UI indication of "stopping for light/sign" (nice-to-have; can mirror existing alerts).

## On-device validation (user, on the comma 4 — cannot be done from the dev box)
1. Build the `.pkl` on-device (Task 6 README); confirm stopmodeld runs and publishes `stopPolicy` at ~10–20 Hz (check with `cereal` tools).
2. `StopPolicyEnabled=1`, drive shadow; review `cloudlog` "stopPolicy SHADOW" lines vs. reality.
3. Only then `StopPolicyActive=1`; supervised, hands ready, on a quiet road; confirm comfortable stops and clean driver override.
