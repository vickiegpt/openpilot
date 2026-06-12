# Stop Policy Training v1 (Plan C — sim + online data) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A working end-to-end training pipeline that trains the distilled-e2e stop policy head on MetaDrive simulator episodes plus online (Hugging Face) detection data, runs a real training on the local GPU, and exports ONNX — no self-collected drives required yet.

**Architecture:** MetaDrive generates scripted traffic-light approach episodes with exact ground-truth labels (the script knows the stop line and light state). A frozen DINOv2 ViT-S/14 encodes frames once into cached features. A step-wise GRU policy head (explicit hidden-state in/out, so ONNX export is streaming-ready) trains on 2-second feature windows with stop-prob/dist/speed losses, plus an auxiliary presence head trained on COCO-streamed traffic-light/stop-sign images. Outputs: metrics CSV, checkpoint, ONNX.

**Tech Stack:** Python 3.x from `/opt/miniconda3` (has torch 2.9.1+cu128, CUDA verified), MetaDrive simulator, HF `datasets` (streaming), DINOv2 via torch.hub, ONNX.

**Parent spec:** `docs/superpowers/specs/2026-06-12-stop-sign-traffic-light-e2e-design.md`

**Hard environment constraints (verified 2026-06-12):**
- Root disk is 100% full: ~8.9 GB free. TOTAL disk budget for this plan: **4 GB** (env ~1.5 GB, sim data ~1 GB, online data ~0.5 GB, weights/caches ~1 GB). Every data-producing task must enforce its byte budget in code.
- GPU: RTX PRO 6000 (96 GB VRAM) — batch sizes are not a concern.
- openpilot's `.venv` does NOT have torch; do not install torch anywhere (reuse miniconda's via `--system-site-packages`).
- CARLA does not fit on disk; MetaDrive is the simulator (openpilot's own sim, see `tools/sim/`).

**Adaptation latitude:** Tasks 2 and 3 call external APIs (MetaDrive, HF datasets) whose exact signatures may differ from the code given here. Implementers MUST keep the interfaces (function signatures, file formats, budgets, label semantics) exactly as specified, but MAY adapt internal API calls to what the installed library actually provides — documenting every adaptation in their report. `tools/sim/bridge/metadrive/metadrive_process.py` is in-repo working MetaDrive camera code to crib from.

---

### Task 0: Training environment

**Files:**
- Create: `tools/stoppolicy/__init__.py` (empty)
- Create: `tools/stoppolicy/setup_env.sh`

- [ ] **Step 1: Write `tools/stoppolicy/setup_env.sh`:**

```bash
#!/usr/bin/env bash
# Training env for the stop policy. Reuses miniconda's torch via system-site-packages
# (disk is nearly full; torch must not be reinstalled).
set -e
VENV=/root/openpilot/tools/stoppolicy/.venv
/opt/miniconda3/bin/python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/pip" install --no-cache-dir "metadrive-simulator>=0.4.2" "datasets>=2.19" pillow onnx pyarrow pytest
"$VENV/bin/python" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible from venv"
print("torch", torch.__version__, "cuda ok")
EOF
```

- [ ] **Step 2: Run it**

Run: `bash tools/stoppolicy/setup_env.sh`
Expected: ends with `torch 2.9.1+cu128 cuda ok` (or similar). If metadrive's pin conflicts with system-site torch deps, install metadrive with `--no-deps` plus its missing deps individually (it mainly needs gymnasium, panda3d, numpy) — document what was needed.

- [ ] **Step 3: Verify DINOv2 loads and disk budget holds**

Run:
```bash
/root/openpilot/tools/stoppolicy/.venv/bin/python -c "
import torch
m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
x = torch.randn(1, 3, 252, 448)
with torch.no_grad():
    out = m.forward_features(x)
print('cls', out['x_norm_clstoken'].shape, 'patch', out['x_norm_patchtokens'].shape)
"
df -h / | tail -1
```
Expected: cls shape (1, 384); disk free not below ~5 GB. Note: DINOv2 requires H,W divisible by 14 — 252x448 is the canonical input for this plan.

- [ ] **Step 4: Commit**

```bash
touch tools/stoppolicy/__init__.py
git add tools/stoppolicy/setup_env.sh tools/stoppolicy/__init__.py
git commit -m "stoppolicy: training environment setup"
```

(Ensure `.venv` is gitignored: add `tools/stoppolicy/.venv/` to `.gitignore` if `git status` shows it.)

---

### Task 1: Episode storage format

One episode = one directory: `frames/%06d.jpg` plus `labels.jsonl` (one JSON object per frame) plus `meta.json`. This is the contract between data generation (Task 2), feature precompute (Task 4), and training (Task 6).

Per-frame label object:
```json
{"idx": 0, "t": 0.0, "speed": 12.3, "stop_prob": 0.0, "dist_to_stop": 87.5,
 "desired_speed": 12.1, "light_state": "green", "has_light": true}
```
- `stop_prob`: 1.0 if the ego is currently obliged to stop for the light ahead (red/yellow and not yet past it), else 0.0
- `dist_to_stop`: meters to the stop line; -1.0 when `stop_prob == 0`
- `desired_speed`: ego speed 1.0 s in the future (the imitation target), m/s
- `light_state`: one of `"red" | "yellow" | "green" | "none"`

**Files:**
- Create: `tools/stoppolicy/data/__init__.py` (empty), `tools/stoppolicy/data/episode.py`
- Test: `tools/stoppolicy/tests/__init__.py` (empty), `tools/stoppolicy/tests/test_episode.py`

All tests in this plan run with: `/root/openpilot/tools/stoppolicy/.venv/bin/pytest tools/stoppolicy/tests/ -v`

- [ ] **Step 1: Write the failing test** — `tools/stoppolicy/tests/test_episode.py`:

```python
import numpy as np

from tools.stoppolicy.data.episode import EpisodeWriter, load_episode


def make_frame(val):
  return np.full((96, 160, 3), val, dtype=np.uint8)


def test_episode_roundtrip(tmp_path):
  ep_dir = tmp_path / "ep0000"
  with EpisodeWriter(ep_dir, source="test", fps=10) as w:
    for i in range(5):
      w.add_frame(make_frame(i * 20), speed=float(i), stop_prob=float(i % 2),
                  dist_to_stop=10.0 * i if i % 2 else -1.0,
                  desired_speed=float(i) + 0.5, light_state="red" if i % 2 else "none")

  ep = load_episode(ep_dir)
  assert ep.meta["source"] == "test"
  assert ep.meta["fps"] == 10
  assert ep.meta["n_frames"] == 5
  assert len(ep.labels) == 5
  assert ep.labels[3]["stop_prob"] == 1.0
  assert ep.labels[3]["dist_to_stop"] == 30.0
  assert ep.labels[3]["light_state"] == "red"
  assert ep.labels[3]["t"] == 0.3
  frames = ep.load_frames()
  assert frames.shape == (5, 96, 160, 3)
  assert frames[2].mean() > frames[0].mean()  # content survived JPEG


def test_writer_enforces_byte_budget(tmp_path):
  with EpisodeWriter(tmp_path / "ep0001", source="test", fps=10, max_bytes=10_000) as w:
    wrote = 0
    for i in range(1000):
      if not w.add_frame(make_frame(i % 255), speed=0.0, stop_prob=0.0,
                         dist_to_stop=-1.0, desired_speed=0.0, light_state="none"):
        break
      wrote += 1
  assert 0 < wrote < 1000  # budget cut it off
```

- [ ] **Step 2: Run, verify FAIL** (ModuleNotFoundError).

- [ ] **Step 3: Write `tools/stoppolicy/data/episode.py`:**

```python
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

LIGHT_STATES = ("none", "green", "yellow", "red")


class EpisodeWriter:
  """Writes one episode: frames/%06d.jpg + labels.jsonl + meta.json.

  add_frame returns False (and stops writing) once max_bytes is exceeded,
  so generators can run open-loop without blowing the disk budget.
  """

  def __init__(self, ep_dir, source, fps, max_bytes=None):
    self.ep_dir = Path(ep_dir)
    (self.ep_dir / "frames").mkdir(parents=True, exist_ok=True)
    self.source = source
    self.fps = fps
    self.max_bytes = max_bytes
    self.bytes_written = 0
    self.n = 0
    self.labels_f = open(self.ep_dir / "labels.jsonl", "w")

  def add_frame(self, rgb, speed, stop_prob, dist_to_stop, desired_speed, light_state):
    assert light_state in LIGHT_STATES
    if self.max_bytes is not None and self.bytes_written >= self.max_bytes:
      return False
    p = self.ep_dir / "frames" / f"{self.n:06d}.jpg"
    Image.fromarray(rgb).save(p, quality=80)
    self.bytes_written += p.stat().st_size
    rec = {"idx": self.n, "t": round(self.n / self.fps, 3), "speed": speed,
           "stop_prob": stop_prob, "dist_to_stop": dist_to_stop,
           "desired_speed": desired_speed, "light_state": light_state,
           "has_light": light_state != "none"}
    self.labels_f.write(json.dumps(rec) + "\n")
    self.n += 1
    return True

  def close(self):
    if self.labels_f is not None:
      self.labels_f.close()
      self.labels_f = None
      meta = {"source": self.source, "fps": self.fps, "n_frames": self.n}
      (self.ep_dir / "meta.json").write_text(json.dumps(meta))

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()


@dataclass
class Episode:
  ep_dir: Path
  meta: dict
  labels: list

  def load_frames(self):
    out = []
    for i in range(self.meta["n_frames"]):
      out.append(np.asarray(Image.open(self.ep_dir / "frames" / f"{i:06d}.jpg")))
    return np.stack(out)


def load_episode(ep_dir):
  ep_dir = Path(ep_dir)
  meta = json.loads((ep_dir / "meta.json").read_text())
  labels = [json.loads(line) for line in (ep_dir / "labels.jsonl").read_text().splitlines()]
  return Episode(ep_dir, meta, labels)
```

- [ ] **Step 4: Run, verify 2 PASSED.**

- [ ] **Step 5: Commit** — `git add tools/stoppolicy/ && git commit -m "stoppolicy: episode storage format"`

---

### Task 2: MetaDrive episode generator

Scripted traffic-light approaches with perfect labels. We do NOT use openpilot's closed-loop bridge; we drive MetaDrive directly with a trivial speed controller because the script knows the stop line and light state — that knowledge IS the label.

**Scenario script per episode:** spawn on a straight PG map; place (or locate) one traffic light ahead at a known longitudinal position; roll at ~12 m/s using a simple proportional speed controller; the light is scripted per-episode as one of: stays green (drive through), red (decelerate to stop 2 m before the stop line, hold 3 s, then green and resume), or no light (control negatives). Labels derive from the script state machine, not from perception.

**Files:**
- Create: `tools/stoppolicy/data/gen_metadrive.py`
- Test: `tools/stoppolicy/tests/test_gen_metadrive.py`

- [ ] **Step 1: Write the failing test** (label logic is pure and testable without the simulator; the sim rollout gets a marked slow test):

```python
import pytest

from tools.stoppolicy.data.gen_metadrive import label_for_state, plan_speed


def test_labels_green_phase():
  rec = label_for_state(ego_x=10.0, speed=12.0, light_x=100.0, light_state="green", future_speed=12.0)
  assert rec["stop_prob"] == 0.0
  assert rec["dist_to_stop"] == -1.0
  assert rec["light_state"] == "green"


def test_labels_red_before_line():
  rec = label_for_state(ego_x=50.0, speed=8.0, light_x=100.0, light_state="red", future_speed=4.0)
  assert rec["stop_prob"] == 1.0
  assert rec["dist_to_stop"] == pytest.approx(48.0)  # stop 2m before the line


def test_labels_red_after_line_is_not_a_stop():
  rec = label_for_state(ego_x=105.0, speed=12.0, light_x=100.0, light_state="red", future_speed=12.0)
  assert rec["stop_prob"] == 0.0


def test_plan_speed_decelerates_to_stop():
  # far away: cruise; near: braking profile; at line: zero
  assert plan_speed(dist_to_stop=80.0, cruise=12.0) == pytest.approx(12.0)
  assert 0.0 < plan_speed(dist_to_stop=10.0, cruise=12.0) < 12.0
  assert plan_speed(dist_to_stop=0.0, cruise=12.0) == 0.0


@pytest.mark.slow
def test_generate_one_episode(tmp_path):
  from tools.stoppolicy.data.gen_metadrive import generate
  eps = generate(out_dir=tmp_path, n_episodes=1, seed=0, max_total_bytes=50_000_000)
  assert len(eps) == 1
  from tools.stoppolicy.data.episode import load_episode
  ep = load_episode(eps[0])
  assert ep.meta["n_frames"] > 50
  states = {r["light_state"] for r in ep.labels}
  assert states & {"red", "green", "none"}
```

- [ ] **Step 2: Run the pure tests, verify FAIL**: `.venv/bin/pytest tools/stoppolicy/tests/test_gen_metadrive.py -v -m "not slow"`

- [ ] **Step 3: Write `tools/stoppolicy/data/gen_metadrive.py`** — required interface and label/control logic:

```python
"""MetaDrive scripted traffic-light episodes with perfect labels.

The episode script (not perception) knows the stop line and light state.
Three scenario kinds, mixed per seed: green-through, red-stop-then-go, no-light.
"""
import numpy as np

from tools.stoppolicy.data.episode import EpisodeWriter

CRUISE = 12.0          # m/s
STOP_MARGIN = 2.0      # stop this many meters before the line
COMFORT_DECEL = 2.0    # m/s^2 used for the braking profile
FPS = 10
IMG_W, IMG_H = 448, 252  # divisible by 14 for DINOv2


def label_for_state(ego_x, speed, light_x, light_state, future_speed):
  must_stop = light_state in ("red", "yellow") and ego_x < light_x
  dist = (light_x - STOP_MARGIN) - ego_x if must_stop else -1.0
  return {"speed": speed, "stop_prob": 1.0 if must_stop else 0.0,
          "dist_to_stop": dist, "desired_speed": future_speed,
          "light_state": light_state if light_x is not None else "none"}


def plan_speed(dist_to_stop, cruise=CRUISE):
  """Target speed obeying a comfortable braking profile toward the stop point."""
  if dist_to_stop <= 0.0:
    return 0.0
  v = (2.0 * COMFORT_DECEL * dist_to_stop) ** 0.5
  return min(cruise, v)


def generate(out_dir, n_episodes, seed, max_total_bytes):
  ...
```

`generate` requirements (implementer fills in the MetaDrive specifics, cribbing from `tools/sim/bridge/metadrive/metadrive_process.py` for env/camera config):
- Headless MetaDrive env, RGB camera at IMG_W x IMG_H, straight-ish PG map, no traffic (`traffic_density=0`).
- Per episode, choose scenario kind from the seed: `green` / `red` / `none` with probabilities 0.4/0.4/0.2.
- Traffic light: locate a PG-map traffic light if the map provides one, else `engine.spawn_object(BaseTrafficLight, lane=...)` at a known lane position ~80–120 m ahead; set its status green or red per scenario (`set_green()` / `set_red()` or status attribute — adapt to installed API). For `red` scenarios: red until ego has been fully stopped for 3 s, then green.
- Drive with simple longitudinal control toward `plan_speed(...)` for red phases and `CRUISE` otherwise (P-controller on speed error → throttle/brake action; steering 0 or lane-keep via MetaDrive's built-in if trivially available).
- Each sim step at FPS: capture RGB frame, compute ego longitudinal position along lane, call `label_for_state`, buffer; `desired_speed`/future_speed is the ego speed FPS steps later — fill by post-processing the buffer (last second repeats final speed) before writing through `EpisodeWriter`.
- Episode length 15–30 s. Respect `max_total_bytes` across all episodes via EpisodeWriter budgets; stop generating when exhausted. Return list of episode dirs.
- `if __name__ == "__main__":` CLI: `--out`, `--n`, `--seed`, `--max-gb` (default 1.0).

- [ ] **Step 4: Pure tests pass**: `-m "not slow"` → 4 PASSED.

- [ ] **Step 5: Slow test passes** (actually boots MetaDrive): `.venv/bin/pytest tools/stoppolicy/tests/test_gen_metadrive.py -v -m slow`. Expect 1 PASSED; first run downloads MetaDrive assets (~300 MB — inside budget). Register the `slow` marker in `tools/stoppolicy/pytest.ini` if pytest warns.

- [ ] **Step 6: Visual sanity artifact** — save a 3x3 contact sheet of frames from a red-scenario episode to `/tmp/stoppolicy_contact_sheet.jpg` (script or snippet is fine) and report whether a traffic light is actually visible in the frames. If MetaDrive's light isn't visibly rendered in the camera, STOP and report DONE_WITH_CONCERNS — visibility is the entire point of the data.

- [ ] **Step 7: Commit** — `git add tools/stoppolicy/ && git commit -m "stoppolicy: MetaDrive scripted traffic-light episode generator"`

---

### Task 3: Online (HF) detection data fetcher

Auxiliary-loss data: single images with `{has_light, has_sign}` presence labels, streamed from COCO on Hugging Face (COCO classes: `traffic light`, `stop sign`). Light *state* supervision comes from sim labels, not COCO.

**Files:**
- Create: `tools/stoppolicy/data/fetch_online.py`
- Test: `tools/stoppolicy/tests/test_fetch_online.py`

Output format: `out_dir/images/%06d.jpg` + `out_dir/labels.jsonl` with `{"idx": i, "has_light": bool, "has_sign": bool}` + `meta.json` `{"n_images": N, "source": "<hf dataset id>"}`.

- [ ] **Step 1: Failing test** (network-free for the core; one marked-slow network test):

```python
import pytest

from tools.stoppolicy.data.fetch_online import presence_from_categories, BudgetedSaver


def test_presence_mapping():
  assert presence_from_categories(["person", "traffic light"]) == (True, False)
  assert presence_from_categories(["stop sign"]) == (False, True)
  assert presence_from_categories(["car", "dog"]) == (False, False)
  assert presence_from_categories(["traffic light", "stop sign"]) == (True, True)


def test_budgeted_saver(tmp_path):
  import numpy as np
  s = BudgetedSaver(tmp_path, max_bytes=20_000)
  n = 0
  while s.add(np.full((128, 128, 3), n % 255, dtype=np.uint8), has_light=bool(n % 2), has_sign=False):
    n += 1
  s.close()
  assert 0 < n < 1000
  meta = __import__("json").loads((tmp_path / "meta.json").read_text())
  assert meta["n_images"] == n


@pytest.mark.slow
def test_fetch_small_sample(tmp_path):
  from tools.stoppolicy.data.fetch_online import fetch
  n = fetch(out_dir=tmp_path, n_target=20, max_bytes=30_000_000)
  assert n >= 10
  labels = [__import__("json").loads(l) for l in (tmp_path / "labels.jsonl").read_text().splitlines()]
  assert any(r["has_light"] or r["has_sign"] for r in labels)
  assert any(not (r["has_light"] or r["has_sign"]) for r in labels)  # negatives kept too
```

- [ ] **Step 2: Run non-slow, verify FAIL.**

- [ ] **Step 3: Implement.** `presence_from_categories(cat_names) -> (has_light, has_sign)` and `BudgetedSaver` are pure (mirror `EpisodeWriter`'s budget pattern). `fetch(out_dir, n_target, max_bytes)` streams a COCO-on-HF dataset (`datasets.load_dataset(..., streaming=True)`); candidate dataset ids to try in order: `detection-datasets/coco`, `rafaelpadilla/coco2017` — verify category names/ids at runtime from the dataset's features rather than hard-coding integers where possible; COCO category ids if needed: traffic light=10, stop sign=13 (1-indexed COCO ids; VERIFY against the chosen dataset's mapping before trusting). Keep a roughly 2:1 positive:negative ratio, resize longest side to 448, respect `max_bytes`, set meta `source` to the dataset id used. CLI: `--out --n --max-gb`.

- [ ] **Step 4: Non-slow tests pass (2), then slow network test passes (1).**

- [ ] **Step 5: Commit** — `git add tools/stoppolicy/ && git commit -m "stoppolicy: HF online detection data fetcher (aux loss)"`

---

### Task 4: Feature precompute

**Files:**
- Create: `tools/stoppolicy/features.py`
- Test: `tools/stoppolicy/tests/test_features.py`

Contract: for an episode dir, write `feats.npy` of shape `(n_frames, 768)` — concat of DINOv2 ViT-S/14 `x_norm_clstoken` (384) and mean of `x_norm_patchtokens` (384), computed at 448x252, ImageNet-normalized, fp16 on disk. For an online-data dir (Task 3 format), same but `(n_images, 768)`.

- [ ] **Step 1: Failing test:**

```python
import numpy as np
import pytest

from tools.stoppolicy.features import FeatureExtractor, precompute_dir


@pytest.fixture(scope="module")
def fx():
  return FeatureExtractor(device="cuda")


@pytest.mark.slow
def test_extractor_shape(fx):
  imgs = np.random.randint(0, 255, (3, 252, 448, 3), dtype=np.uint8)
  f = fx.encode(imgs)
  assert f.shape == (3, 768)
  assert f.dtype == np.float16


@pytest.mark.slow
def test_precompute_episode(fx, tmp_path):
  from tools.stoppolicy.data.episode import EpisodeWriter
  with EpisodeWriter(tmp_path / "ep", source="test", fps=10) as w:
    for i in range(4):
      w.add_frame(np.random.randint(0, 255, (252, 448, 3), dtype=np.uint8),
                  speed=0.0, stop_prob=0.0, dist_to_stop=-1.0, desired_speed=0.0, light_state="none")
  precompute_dir(tmp_path / "ep", fx)
  f = np.load(tmp_path / "ep" / "feats.npy")
  assert f.shape == (4, 768)
```

- [ ] **Step 2: Verify FAIL, then implement** `features.py`: `FeatureExtractor` loads `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`, eval+no_grad, `encode(uint8 NHWC) -> (N,768) fp16` with batching (batch 64), resize to 448x252 if needed, ImageNet mean/std. `precompute_dir(path, fx)` handles both episode dirs (frames/) and online dirs (images/), writes `feats.npy`. CLI: `--data-root` precomputes every episode/online dir under a root, skipping dirs that already have `feats.npy`.

- [ ] **Step 3: Slow tests pass (2). Commit** — `"stoppolicy: DINOv2 feature precompute"`

---

### Task 5: Policy model

**Files:**
- Create: `tools/stoppolicy/model.py`
- Test: `tools/stoppolicy/tests/test_model.py`

Step-wise (streaming-ready) design: the core is a GRUCell over per-frame features; hidden state is an explicit input/output so the same module trains (loop over T) and exports to ONNX as a single-step model.

- [ ] **Step 1: Failing test:**

```python
import torch

from tools.stoppolicy.model import StopPolicyHead, AuxPresenceHead, FEAT_DIM, HIDDEN_DIM


def test_policy_step_shapes():
  m = StopPolicyHead()
  feat = torch.randn(8, FEAT_DIM)
  speed = torch.randn(8, 1)
  h = m.initial_hidden(8)
  out, h2 = m.step(feat, speed, h)
  assert out["stop_logit"].shape == (8, 1)
  assert out["dist_to_stop"].shape == (8, 1)
  assert out["desired_speed"].shape == (8, 1)
  assert h2.shape == (8, HIDDEN_DIM)


def test_policy_sequence_matches_steps():
  m = StopPolicyHead().eval()
  feats = torch.randn(2, 5, FEAT_DIM)
  speeds = torch.randn(2, 5, 1)
  with torch.no_grad():
    seq_out = m.forward_sequence(feats, speeds)
    h = m.initial_hidden(2)
    for t in range(5):
      step_out, h = m.step(feats[:, t], speeds[:, t], h)
    assert torch.allclose(seq_out["stop_logit"][:, -1], step_out["stop_logit"], atol=1e-5)


def test_aux_head_shapes():
  a = AuxPresenceHead()
  logits = a(torch.randn(4, FEAT_DIM))
  assert logits.shape == (4, 2)  # has_light, has_sign
```

- [ ] **Step 2: Verify FAIL, then implement `model.py`:**

```python
import torch
import torch.nn as nn

FEAT_DIM = 768
HIDDEN_DIM = 256


class StopPolicyHead(nn.Module):
  """Step-wise GRU policy head. Explicit hidden state -> exports to ONNX as a
  single-step model and streams on-device; forward_sequence is the training path."""

  def __init__(self):
    super().__init__()
    self.inp = nn.Sequential(nn.Linear(FEAT_DIM + 1, 512), nn.ReLU(), nn.Linear(512, 256))
    self.cell = nn.GRUCell(256, HIDDEN_DIM)
    self.stop_head = nn.Linear(HIDDEN_DIM, 1)
    self.dist_head = nn.Linear(HIDDEN_DIM, 1)
    self.speed_head = nn.Linear(HIDDEN_DIM, 1)

  def initial_hidden(self, batch):
    return torch.zeros(batch, HIDDEN_DIM, device=next(self.parameters()).device)

  def step(self, feat, speed, hidden):
    x = self.inp(torch.cat([feat, speed], dim=-1))
    h = self.cell(x, hidden)
    out = {"stop_logit": self.stop_head(h),
           "dist_to_stop": torch.nn.functional.softplus(self.dist_head(h)),
           "desired_speed": torch.nn.functional.softplus(self.speed_head(h))}
    return out, h

  def forward_sequence(self, feats, speeds):
    B, T, _ = feats.shape
    h = self.initial_hidden(B)
    outs = {"stop_logit": [], "dist_to_stop": [], "desired_speed": []}
    for t in range(T):
      out, h = self.step(feats[:, t], speeds[:, t], h)
      for k in outs:
        outs[k].append(out[k])
    return {k: torch.stack(v, dim=1) for k, v in outs.items()}


class AuxPresenceHead(nn.Module):
  """has_light / has_sign presence from a single frame's features (aux loss)."""

  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(FEAT_DIM, 256), nn.ReLU(), nn.Linear(256, 2))

  def forward(self, feat):
    return self.net(feat)
```

- [ ] **Step 3: 3 PASSED. Commit** — `"stoppolicy: step-wise GRU policy head + aux presence head"`

---

### Task 6: Training loop + metrics

**Files:**
- Create: `tools/stoppolicy/train.py`
- Test: `tools/stoppolicy/tests/test_train.py`

Design:
- `WindowDataset`: 2 s windows (20 frames @ 10 Hz) over episode feats+labels; episode-level train/val split (no window leakage across the split); oversample windows containing `stop_prob == 1` to ~50%.
- Losses: BCE-with-logits on stop (per step), L1 on `dist_to_stop` masked to `stop_prob == 1` steps (normalize: dist/100), L1 on `desired_speed` (normalize: /CRUISE≈12), aux BCE on presence from online-data feats (interleaved batches), weights 1.0 / 0.5 / 0.5 / 0.2.
- Eval on val episodes, full-sequence streaming: **stop recall** (fraction of red-phase windows where max sigmoid(stop_logit) > 0.5), **false-stop rate** (fraction of no-stop windows where it fires), **dist MAE** (m, on true-stop steps), written to `runs/<name>/metrics.csv` per epoch; best checkpoint by (recall - false_rate) saved to `runs/<name>/best.pt` (state_dict of both heads + a json of normalization constants).
- AdamW lr 3e-4, cosine decay, fp32 (head is tiny), `--epochs --data-root --online-root --run-name --seed` CLI. Deterministic seeding.

- [ ] **Step 1: Failing test** (tiny synthetic feats, no GPU dependency — model runs on CPU here):

```python
import json

import numpy as np

from tools.stoppolicy.train import WindowDataset, train


def synth_episode(tmp_path, name, red):
  """20s episode: red ones decelerate mid-episode; feats are label-correlated so
  the model can actually learn — channel 0 carries the stop signal."""
  from tools.stoppolicy.data.episode import EpisodeWriter
  ep = tmp_path / name
  T = 200
  with EpisodeWriter(ep, source="synth", fps=10) as w:
    for i in range(T):
      stopping = red and 50 <= i < 150
      w.add_frame(np.zeros((28, 28, 3), dtype=np.uint8), speed=12.0 if not stopping else 2.0,
                  stop_prob=1.0 if stopping else 0.0,
                  dist_to_stop=max(0.0, 100.0 - i) if stopping else -1.0,
                  desired_speed=2.0 if stopping else 12.0,
                  light_state="red" if stopping else "none")
  feats = np.zeros((T, 768), dtype=np.float16)
  for i in range(T):
    feats[i, 0] = 1.0 if (red and 50 <= i < 150) else -1.0
  feats += np.random.randn(T, 768).astype(np.float16) * 0.01
  np.save(ep / "feats.npy", feats)
  return ep


def test_window_dataset_split_and_oversample(tmp_path):
  eps = [synth_episode(tmp_path, f"ep{i}", red=i % 2 == 0) for i in range(6)]
  ds = WindowDataset([str(e) for e in eps], window=20, oversample_stops=True, seed=0)
  feats, speeds, targets = ds[0]
  assert feats.shape == (20, 768)
  assert speeds.shape == (20, 1)
  assert targets["stop_prob"].shape == (20,)
  frac_stop = np.mean([ds[i][2]["stop_prob"].max() for i in range(len(ds))])
  assert frac_stop > 0.3  # oversampled


def test_train_learns_synthetic(tmp_path):
  for i in range(8):
    synth_episode(tmp_path / "data", f"ep{i}", red=i % 2 == 0)
  metrics = train(data_root=tmp_path / "data", online_root=None, run_dir=tmp_path / "run",
                  epochs=8, seed=0, device="cpu")
  assert metrics["stop_recall"] > 0.8
  assert metrics["false_stop_rate"] < 0.2
  assert (tmp_path / "run" / "best.pt").exists()
  assert (tmp_path / "run" / "metrics.csv").exists()
  norm = json.loads((tmp_path / "run" / "norm.json").read_text())
  assert "dist_scale" in norm and "speed_scale" in norm
```

- [ ] **Step 2: Verify FAIL, implement `train.py`** per the design above. `train(...)` returns the final-epoch val metrics dict. Keep it under ~250 lines; no tensorboard, just CSV + prints.

- [ ] **Step 3: Tests pass (the learning test must genuinely pass — if it's flaky, fix the model/lr, don't loosen thresholds). Commit** — `"stoppolicy: training loop with stop-recall/false-stop metrics"`

---

### Task 7: ONNX export

**Files:**
- Create: `tools/stoppolicy/export.py`
- Test: `tools/stoppolicy/tests/test_export.py`

- [ ] **Step 1: Failing test:**

```python
import numpy as np
import torch

from tools.stoppolicy.export import export_onnx
from tools.stoppolicy.model import StopPolicyHead, FEAT_DIM, HIDDEN_DIM


def test_export_and_parity(tmp_path):
  m = StopPolicyHead().eval()
  path = tmp_path / "stop_policy.onnx"
  export_onnx(m, path)

  import onnxruntime as ort
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  feat = np.random.randn(1, FEAT_DIM).astype(np.float32)
  speed = np.random.randn(1, 1).astype(np.float32)
  h = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
  ort_out = sess.run(None, {"feat": feat, "speed": speed, "hidden_in": h})
  with torch.no_grad():
    t_out, t_h = m.step(torch.from_numpy(feat), torch.from_numpy(speed), torch.from_numpy(h))
  np.testing.assert_allclose(ort_out[0], t_out["stop_logit"].numpy(), atol=1e-4)
  np.testing.assert_allclose(ort_out[3], t_h.numpy(), atol=1e-4)
```

- [ ] **Step 2: Verify FAIL, implement `export.py`:** wrap `step` in a `nn.Module` whose forward takes `(feat, speed, hidden_in)` and returns `(stop_logit, dist_to_stop, desired_speed, hidden_out)`; `torch.onnx.export` with those input/output names, opset 17. `export_onnx(model, path)` plus CLI `--checkpoint --out`. If `onnxruntime` isn't installed, `pip install --no-cache-dir onnxruntime` into the training venv (CPU build, small) and note it in setup_env.sh too.

- [ ] **Step 3: 1 PASSED. Commit** — `"stoppolicy: ONNX export (streaming single-step)"`

---

### Task 8: The actual training run (sim + online data)

No new source files — this task RUNS the pipeline on the GPU and records results. Budget guardrail: check `df -h /` before each stage; abort below 3 GB free.

- [ ] **Step 1: Generate sim data** — `.venv/bin/python -m tools.stoppolicy.data.gen_metadrive --out /root/stoppolicy_data/sim --n 60 --seed 1 --max-gb 1.0` (~60 episodes ≈ 20–30 min of driving).
- [ ] **Step 2: Fetch online data** — `.venv/bin/python -m tools.stoppolicy.data.fetch_online --out /root/stoppolicy_data/online --n 2000 --max-gb 0.5`
- [ ] **Step 3: Precompute features** — `.venv/bin/python -m tools.stoppolicy.features --data-root /root/stoppolicy_data`
- [ ] **Step 4: Train** — `.venv/bin/python -m tools.stoppolicy.train --data-root /root/stoppolicy_data/sim --online-root /root/stoppolicy_data/online --run-name v1 --epochs 40 --seed 0`
- [ ] **Step 5: Export** — `.venv/bin/python -m tools.stoppolicy.export --checkpoint runs/v1/best.pt --out runs/v1/stop_policy.onnx`
- [ ] **Step 6: Record results** — append a `## v1 results (sim+online)` section to `tools/stoppolicy/README.md` (create it: usage of each stage + this results table): stop recall, false-stop rate, dist MAE on val episodes, n episodes/images, disk used. Commit code+README (NOT the data/runs; gitignore `runs/` and any data dirs).

Success bar for v1 (sim-only validation): stop recall ≥ 0.9, false-stop rate ≤ 0.1, dist MAE ≤ 10 m on held-out sim episodes. If unmet, one iteration pass on hyperparameters/data volume is in scope; deeper model changes are not (report instead).

---

## Out of scope (later plans / after real-drive data exists)

- Real-drive auto-labeling (needs the Plan A rig footage)
- Light-state (red/yellow/green) aux supervision from online data with state labels
- stopmodeld/on-device integration (Plan B), CARLA (disk), multi-camera fusion (sim has one camera; the head's FEAT_DIM contract is where telephoto features will concatenate later — that will be a deliberate schema change)
