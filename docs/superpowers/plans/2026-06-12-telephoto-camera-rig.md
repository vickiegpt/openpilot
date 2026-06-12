# Telephoto Camera Rig (Plan A of Stop Policy Project) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two USB telephoto cameras stream into openpilot via VisionIPC, publish a logged `telephotoCameraState` message, and record H.264 segments for training data — the spec's Phase 1 "rig bring-up" gate.

**Architecture:** A new `telecamerad` daemon (pattern: `tools/webcam/camerad.py`) runs its own VisionIPC server named `"telecamerad"`, reusing existing `VisionStreamType` slots so the msgq submodule is never forked. Frame metadata goes out as `telephotoCameraState` (a renamed `CustomReserved0` slot — the sanctioned fork mechanism in `cereal/custom.capnp`). Because `encoderd` only encodes the built-in cameras, `telecamerad` records its own H.264 mp4 segments with a frameId sidecar for the training pipeline.

**Tech Stack:** Python, cereal/msgq (capnp, VisionIPC), OpenCV + PyAV (existing webcam deps), pytest.

**Parent spec:** `docs/superpowers/specs/2026-06-12-stop-sign-traffic-light-e2e-design.md`

**Follow-on plans (not in this document):**
- Plan B: `stopmodeld` runner (eGPU) + `stopPolicy` message + longitudinal planner integration + shadow mode
- Plan C: offline training pipeline (auto-labeling, dataset, training, ONNX export)
- Plan D: CARLA closed-loop evaluation suite

---

### Task 0: Environment bring-up

This checkout may have uninitialized submodules. Nothing in later tasks works without msgq/cereal built.

**Files:** none modified.

- [ ] **Step 1: Initialize submodules**

```bash
cd /root/openpilot
git submodule update --init --recursive msgq_repo opendbc_repo rednose_repo panda tinygrad_repo
```

- [ ] **Step 2: Build**

```bash
scons -j$(nproc) cereal msgq_repo
```

If that target spelling fails, fall back to a full `scons -j$(nproc)`.
Expected: exits 0.

- [ ] **Step 3: Smoke-check the Python imports the plan depends on**

```bash
python3 -c "from msgq.visionipc import VisionIpcServer, VisionStreamType; from cereal import messaging; import av, cv2; print('ok')"
```

Expected output: `ok`. Do not proceed until this passes.

---

### Task 1: `telephotoCameraState` cereal message

**Files:**
- Modify: `cereal/custom.capnp` (struct `CustomReserved0`)
- Modify: `cereal/log.capnp:2551` (union field `customReserved0`)
- Modify: `cereal/services.py` (add service)
- Create: `selfdrive/telecamerad/__init__.py` (empty)
- Create: `selfdrive/telecamerad/tests/__init__.py` (empty)
- Test: `selfdrive/telecamerad/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
from cereal import messaging
from cereal.services import SERVICE_LIST


def test_telephoto_camera_state_roundtrip():
  dat = messaging.new_message('telephotoCameraState', valid=True)
  dat.telephotoCameraState.frameId = 42
  dat.telephotoCameraState.cameraIndex = 1
  dat.telephotoCameraState.timestampEof = 123456789
  evt = messaging.log_from_bytes(dat.to_bytes())
  assert evt.telephotoCameraState.frameId == 42
  assert evt.telephotoCameraState.cameraIndex == 1
  assert evt.telephotoCameraState.timestampEof == 123456789


def test_telephoto_camera_state_service():
  svc = SERVICE_LIST['telephotoCameraState']
  assert svc.should_log
  assert svc.frequency == 40.  # 2 cameras x 20Hz on one service
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest selfdrive/telecamerad/tests/test_schema.py -v`
Expected: FAIL (capnp has no `telephotoCameraState` field / KeyError in SERVICE_LIST).

- [ ] **Step 3: Rename the reserved struct in `cereal/custom.capnp`**

Replace:

```capnp
struct CustomReserved0 @0x81c2f05a394cf4af {
}
```

with (same identifier — the `@0x...` must NOT change):

```capnp
struct TelephotoCameraState @0x81c2f05a394cf4af {
  frameId @0 :UInt32;
  cameraIndex @1 :UInt8;   # 0 or 1: which telephoto camera
  timestampEof @2 :UInt64; # ns, end of frame capture
}
```

- [ ] **Step 4: Rename the union field in `cereal/log.capnp`**

Replace line 2551:

```capnp
    customReserved0 @107 :Custom.CustomReserved0;
```

with (the `@107` must NOT change):

```capnp
    telephotoCameraState @107 :Custom.TelephotoCameraState;
```

- [ ] **Step 5: Register the service in `cereal/services.py`**

In the `_services` dict, directly below the line `"driverCameraState": (True, 20., 20),` add:

```python
  "telephotoCameraState": (True, 40., 40),  # 2 telephoto cams x 20Hz, fork service (CustomReserved0)
```

- [ ] **Step 6: Create the empty package files**

```bash
touch selfdrive/telecamerad/__init__.py selfdrive/telecamerad/tests/__init__.py
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest selfdrive/telecamerad/tests/test_schema.py -v`
Expected: 2 PASSED.

- [ ] **Step 8: Run the cereal consistency tests to catch schema/service mismatches**

Run: `pytest cereal/ -x -q`
Expected: PASS (no failures introduced).

- [ ] **Step 9: Commit**

```bash
git add cereal/custom.capnp cereal/log.capnp cereal/services.py selfdrive/telecamerad/
git commit -m "cereal: add telephotoCameraState fork message (CustomReserved0)"
```

---

### Task 2: Parameterize the webcam `Camera` class

`tools/webcam/camera.py` hardcodes 1280x720@25 and a 180° flip. The telephoto cameras need configurable resolution/fps and no flip. Parameterize with backwards-compatible defaults so `webcamerad` is unchanged.

**Files:**
- Modify: `tools/webcam/camera.py`
- Test: `selfdrive/telecamerad/tests/test_camera.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import cv2 as cv

import openpilot.tools.webcam.camera as camera_mod
from openpilot.tools.webcam.camera import Camera


class FakeCap:
  def __init__(self, frames):
    self.frames = list(frames)
    self.props = {}

  def set(self, prop, val):
    self.props[prop] = val

  def get(self, prop):
    return self.props.get(prop, 0.0)

  def read(self):
    if self.frames:
      return True, self.frames.pop(0)
    return False, None

  def release(self):
    pass


def make_bgr(w=64, h=48):
  bgr = np.zeros((h, w, 3), dtype=np.uint8)
  bgr[0, 0] = (255, 255, 255)  # bright corner marker
  return bgr


def test_camera_applies_params(monkeypatch):
  cap = FakeCap([])
  monkeypatch.setattr(camera_mod.cv, "VideoCapture", lambda _id: cap)
  Camera("telephotoCameraState", 0, "9", width=1920, height=1080, fps=20, rotate_180=False)
  assert cap.props[cv.CAP_PROP_FRAME_WIDTH] == 1920.0
  assert cap.props[cv.CAP_PROP_FRAME_HEIGHT] == 1080.0
  assert cap.props[cv.CAP_PROP_FPS] == 20.0


def test_camera_defaults_unchanged(monkeypatch):
  # webcamerad's existing behavior must not change
  cap = FakeCap([])
  monkeypatch.setattr(camera_mod.cv, "VideoCapture", lambda _id: cap)
  cam = Camera("roadCameraState", 0, "0")
  assert cap.props[cv.CAP_PROP_FRAME_WIDTH] == 1280.0
  assert cap.props[cv.CAP_PROP_FPS] == 25.0
  assert cam.rotate_180 is True


def test_read_frames_no_rotate(monkeypatch):
  bgr = make_bgr()
  cap = FakeCap([bgr.copy()])
  monkeypatch.setattr(camera_mod.cv, "VideoCapture", lambda _id: cap)
  cam = Camera("telephotoCameraState", 0, "9", rotate_180=False)
  frames = list(cam.read_frames())
  assert len(frames) == 1
  assert frames[0] == Camera.bgr2nv12(bgr).data.tobytes()  # unflipped


def test_read_frames_rotate(monkeypatch):
  bgr = make_bgr()
  cap = FakeCap([bgr.copy()])
  monkeypatch.setattr(camera_mod.cv, "VideoCapture", lambda _id: cap)
  cam = Camera("roadCameraState", 0, "0")  # default rotate_180=True
  frames = list(cam.read_frames())
  assert frames[0] == Camera.bgr2nv12(cv.flip(bgr, -1)).data.tobytes()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest selfdrive/telecamerad/tests/test_camera.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'width'`.

- [ ] **Step 3: Replace `tools/webcam/camera.py` with the parameterized version**

```python
import av
import cv2 as cv

class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id, width=1280, height=720, fps=25, rotate_180=True):
    try:
      camera_id = int(camera_id)
    except ValueError: # allow strings, ex: /dev/video0
      pass
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0
    self.rotate_180 = rotate_180

    print(f"Opening {cam_type_state} at {camera_id}")

    self.cap = cv.VideoCapture(camera_id)

    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, float(width))
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, float(height))
    self.cap.set(cv.CAP_PROP_FPS, float(fps))

    self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
    self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)

  @classmethod
  def bgr2nv12(self, bgr):
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray()

  def read_frames(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if self.rotate_180:
        frame = cv.flip(frame, -1)
      yuv = Camera.bgr2nv12(frame)
      yield yuv.data.tobytes()
    self.cap.release()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest selfdrive/telecamerad/tests/test_camera.py -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add tools/webcam/camera.py selfdrive/telecamerad/tests/test_camera.py
git commit -m "webcam: parameterize Camera resolution/fps/rotation"
```

---

### Task 3: `telecamerad` daemon

**Files:**
- Create: `selfdrive/telecamerad/telecamerad.py`
- Test: `selfdrive/telecamerad/tests/test_telecamerad.py`

Design note: cameras are injectable so tests run with fakes and no /dev/video devices. The two camera threads share one `PubMaster` publishing the same service, so `_send` takes a lock (webcamerad doesn't need one because each of its threads owns a distinct service).

- [ ] **Step 1: Write the failing test**

```python
import time

import numpy as np

from cereal import messaging
from openpilot.selfdrive.telecamerad.telecamerad import TeleCamerad, STREAM_SLOTS

W, H = 64, 48
N_FRAMES = 5


class FakeCamera:
  def __init__(self, n_frames=N_FRAMES):
    self.W, self.H = float(W), float(H)
    self.cur_frame_id = 0
    self.n_frames = n_frames

  def read_frames(self):
    nv12 = np.zeros((H * 3 // 2, W), dtype=np.uint8)
    for _ in range(self.n_frames):
      yield nv12.tobytes()


def drain(sock, expected, timeout_s=5.0):
  msgs = []
  end = time.monotonic() + timeout_s
  while len(msgs) < expected and time.monotonic() < end:
    msgs.extend(messaging.drain_sock(sock, wait_for_one=False))
    time.sleep(0.01)
  return msgs


def test_single_camera_publishes_states():
  sock = messaging.sub_sock('telephotoCameraState', conflate=False, timeout=100)
  tcd = TeleCamerad(cameras=[FakeCamera()])
  time.sleep(0.2)  # let the subscriber connect
  tcd.camera_runner(0, tcd.cameras[0])
  msgs = drain(sock, N_FRAMES)
  assert len(msgs) == N_FRAMES
  assert [m.telephotoCameraState.frameId for m in msgs] == list(range(N_FRAMES))
  assert all(m.telephotoCameraState.cameraIndex == 0 for m in msgs)


def test_two_cameras_run_to_completion():
  sock = messaging.sub_sock('telephotoCameraState', conflate=False, timeout=100)
  tcd = TeleCamerad(cameras=[FakeCamera(), FakeCamera()])
  time.sleep(0.2)
  tcd.run()  # camera generators are finite, so this returns
  msgs = drain(sock, 2 * N_FRAMES)
  by_cam = {0: [], 1: []}
  for m in msgs:
    by_cam[m.telephotoCameraState.cameraIndex].append(m.telephotoCameraState.frameId)
  assert by_cam[0] == list(range(N_FRAMES))
  assert by_cam[1] == list(range(N_FRAMES))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest selfdrive/telecamerad/tests/test_telecamerad.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'openpilot.selfdrive.telecamerad.telecamerad'`.

- [ ] **Step 3: Write `selfdrive/telecamerad/telecamerad.py`**

```python
#!/usr/bin/env python3
import os
import threading

from msgq.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.common.realtime import Ratekeeper
from openpilot.tools.webcam.camera import Camera

# The telephoto cameras get their own VisionIPC server ("telecamerad").
# Stream types only need to be unique per server, so we reuse the ROAD and
# WIDE_ROAD slots instead of forking msgq to add new enum values. Clients
# must connect with the server name "telecamerad" — on this server these
# slots mean telephoto cam 0 and 1, not the built-in cameras.
STREAM_SLOTS = [VisionStreamType.VISION_STREAM_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD]
FPS = 20


def env_cameras():
  cams = []
  for idx in (0, 1):
    dev = os.getenv(f"TELEPHOTO_CAM_{idx}")
    if dev is not None:
      cams.append(Camera("telephotoCameraState", STREAM_SLOTS[idx], dev,
                         width=int(os.getenv("TELEPHOTO_WIDTH", "1920")),
                         height=int(os.getenv("TELEPHOTO_HEIGHT", "1080")),
                         fps=FPS, rotate_180=False))
  return cams


class TeleCamerad:
  def __init__(self, cameras=None):
    self.cameras = cameras if cameras is not None else env_cameras()
    assert len(self.cameras) > 0, "set TELEPHOTO_CAM_0 (and optionally TELEPHOTO_CAM_1)"

    self.pm = messaging.PubMaster(["telephotoCameraState"])
    self.pm_lock = threading.Lock()

    self.vipc_server = VisionIpcServer("telecamerad")
    for idx, cam in enumerate(self.cameras):
      self.vipc_server.create_buffers(STREAM_SLOTS[idx], 20, int(cam.W), int(cam.H))
    self.vipc_server.start_listener()

  def _send(self, idx, cam, yuv):
    eof = int(cam.cur_frame_id * (1 / FPS) * 1e9)
    self.vipc_server.send(STREAM_SLOTS[idx], yuv, cam.cur_frame_id, eof, eof)
    dat = messaging.new_message("telephotoCameraState", valid=True)
    dat.telephotoCameraState.frameId = cam.cur_frame_id
    dat.telephotoCameraState.cameraIndex = idx
    dat.telephotoCameraState.timestampEof = eof
    with self.pm_lock:
      self.pm.send("telephotoCameraState", dat)

  def camera_runner(self, idx, cam):
    rk = Ratekeeper(FPS, None)
    for yuv in cam.read_frames():
      self._send(idx, cam, yuv)
      cam.cur_frame_id += 1
      rk.keep_time()

  def run(self):
    threads = [threading.Thread(target=self.camera_runner, args=(i, c)) for i, c in enumerate(self.cameras)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()


def main():
  TeleCamerad().run()


if __name__ == "__main__":
  main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest selfdrive/telecamerad/tests/test_telecamerad.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add selfdrive/telecamerad/telecamerad.py selfdrive/telecamerad/tests/test_telecamerad.py
git commit -m "telecamerad: telephoto camera daemon with VisionIPC + telephotoCameraState"
```

---

### Task 4: Training-data recorder

`encoderd` only encodes the built-in cameras, so `telecamerad` records its own H.264 segments. Each segment is an mp4 plus a `.jsonl` sidecar mapping mp4 frame order to `frameId`/`timestampEof`, which the training pipeline (Plan C) joins against the rlogs.

**Files:**
- Create: `selfdrive/telecamerad/recorder.py`
- Modify: `selfdrive/telecamerad/telecamerad.py` (wire in recording)
- Test: `selfdrive/telecamerad/tests/test_recorder.py`

- [ ] **Step 1: Write the failing test**

```python
import json

import av
import numpy as np

from openpilot.selfdrive.telecamerad.recorder import TeleRecorder

W, H = 64, 48


def nv12_frame(val):
  return np.full((H * 3 // 2, W), val, dtype=np.uint8).tobytes()


def count_mp4_frames(path):
  with av.open(str(path)) as container:
    return sum(1 for _ in container.decode(video=0))


def test_recorder_segments_and_sidecar(tmp_path):
  rec = TeleRecorder(tmp_path, camera_index=0, width=W, height=H, fps=20, segment_len_frames=3)
  for i in range(5):
    rec.write(nv12_frame(i * 10), frame_id=i, timestamp_eof=i * 50_000_000)
  rec.close()

  seg0_mp4 = tmp_path / "cam0_seg0000.mp4"
  seg1_mp4 = tmp_path / "cam0_seg0001.mp4"
  assert seg0_mp4.exists() and seg1_mp4.exists()
  assert count_mp4_frames(seg0_mp4) == 3
  assert count_mp4_frames(seg1_mp4) == 2

  sidecar0 = [json.loads(l) for l in (tmp_path / "cam0_seg0000.jsonl").read_text().splitlines()]
  sidecar1 = [json.loads(l) for l in (tmp_path / "cam0_seg0001.jsonl").read_text().splitlines()]
  assert [s["frameId"] for s in sidecar0] == [0, 1, 2]
  assert [s["frameId"] for s in sidecar1] == [3, 4]
  assert sidecar1[0]["timestampEof"] == 3 * 50_000_000


def test_recorder_close_is_idempotent(tmp_path):
  rec = TeleRecorder(tmp_path, camera_index=1, width=W, height=H)
  rec.write(nv12_frame(0), frame_id=0, timestamp_eof=0)
  rec.close()
  rec.close()  # must not raise
  assert (tmp_path / "cam1_seg0000.mp4").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest selfdrive/telecamerad/tests/test_recorder.py -v`
Expected: FAIL with `ModuleNotFoundError` for `recorder`.

- [ ] **Step 3: Write `selfdrive/telecamerad/recorder.py`**

```python
import json
from pathlib import Path

import av
import numpy as np


class TeleRecorder:
  """H.264 mp4 segments + a frameId sidecar (.jsonl) for one telephoto camera.

  encoderd only handles the built-in cameras, so telecamerad records its own
  training footage. The sidecar maps mp4 frame order to frameId/timestampEof
  so the training pipeline can join footage against rlogs.
  """

  def __init__(self, out_dir, camera_index, width, height, fps=20, segment_len_frames=1200):
    self.out_dir = Path(out_dir)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.camera_index = camera_index
    self.w, self.h, self.fps = width, height, fps
    self.segment_len_frames = segment_len_frames

    self.container = None
    self.sidecar = None
    self.seg_idx = -1
    self.frames_in_seg = 0
    self.pts = 0

  def _open_segment(self):
    self._close_segment()
    self.seg_idx += 1
    base = self.out_dir / f"cam{self.camera_index}_seg{self.seg_idx:04d}"
    self.container = av.open(str(base.with_suffix(".mp4")), mode="w")
    self.stream = self.container.add_stream("h264", rate=self.fps)
    self.stream.width, self.stream.height = self.w, self.h
    self.stream.pix_fmt = "yuv420p"
    self.sidecar = open(base.with_suffix(".jsonl"), "w")
    self.frames_in_seg = 0
    self.pts = 0

  def _close_segment(self):
    if self.container is not None:
      for pkt in self.stream.encode():  # flush
        self.container.mux(pkt)
      self.container.close()
      self.container = None
    if self.sidecar is not None:
      self.sidecar.close()
      self.sidecar = None

  def write(self, nv12_bytes, frame_id, timestamp_eof):
    if self.container is None or self.frames_in_seg >= self.segment_len_frames:
      self._open_segment()
    arr = np.frombuffer(nv12_bytes, dtype=np.uint8).reshape(self.h * 3 // 2, self.w)
    frame = av.VideoFrame.from_ndarray(arr, format="nv12").reformat(format="yuv420p")
    frame.pts = self.pts
    self.pts += 1
    for pkt in self.stream.encode(frame):
      self.container.mux(pkt)
    self.sidecar.write(json.dumps({"frameId": frame_id, "timestampEof": timestamp_eof}) + "\n")
    self.sidecar.flush()
    self.frames_in_seg += 1

  def close(self):
    self._close_segment()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest selfdrive/telecamerad/tests/test_recorder.py -v`
Expected: 2 PASSED.

- [ ] **Step 5: Wire recording into the daemon**

In `selfdrive/telecamerad/telecamerad.py`, add to the imports:

```python
from openpilot.selfdrive.telecamerad.recorder import TeleRecorder
```

In `TeleCamerad.__init__`, after `start_listener()`:

```python
    record_dir = os.getenv("TELEPHOTO_RECORD_DIR")
    self.recorders = {}
    if record_dir is not None:
      for idx, cam in enumerate(self.cameras):
        self.recorders[idx] = TeleRecorder(record_dir, idx, int(cam.W), int(cam.H), fps=FPS)
```

In `_send`, after `self.vipc_server.send(...)`:

```python
    if idx in self.recorders:
      self.recorders[idx].write(yuv, cam.cur_frame_id, eof)
```

At the end of `run()`, after the join loop:

```python
    for rec in self.recorders.values():
      rec.close()
```

- [ ] **Step 6: Add a recording test to `selfdrive/telecamerad/tests/test_telecamerad.py`**

```python
def test_recording_enabled(tmp_path, monkeypatch):
  monkeypatch.setenv("TELEPHOTO_RECORD_DIR", str(tmp_path))
  tcd = TeleCamerad(cameras=[FakeCamera()])
  tcd.run()
  assert (tmp_path / "cam0_seg0000.mp4").exists()
  sidecar = (tmp_path / "cam0_seg0000.jsonl").read_text().splitlines()
  assert len(sidecar) == N_FRAMES
```

- [ ] **Step 7: Run all telecamerad tests**

Run: `pytest selfdrive/telecamerad/tests/ -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add selfdrive/telecamerad/
git commit -m "telecamerad: record H.264 segments with frameId sidecar for training data"
```

---

### Task 5: Manager integration

**Files:**
- Modify: `system/manager/process_config.py`

- [ ] **Step 1: Add the env flag and process entry**

In `system/manager/process_config.py`, below the line `WEBCAM = os.getenv("USE_WEBCAM") is not None` (line 10), add:

```python
TELECAM = os.getenv("TELEPHOTO_CAM_0") is not None
```

In the `procs` list, below the `webcamerad` entry (line 76), add:

```python
  PythonProcess("telecamerad", "selfdrive.telecamerad.telecamerad", only_onroad, enabled=TELECAM),
```

- [ ] **Step 2: Verify the config still parses and existing manager tests pass**

Run: `python3 -c "from openpilot.system.manager.process_config import procs; assert any(p.name == 'telecamerad' for p in procs); print('ok')"`
Expected: `ok`

Run: `pytest system/manager/tests/ -x -q`
Expected: PASS (no failures introduced).

- [ ] **Step 3: Commit**

```bash
git add system/manager/process_config.py
git commit -m "manager: start telecamerad when TELEPHOTO_CAM_0 is set"
```

---

### Task 6: Rig bring-up check tool

A CLI that watches `telephotoCameraState` for N seconds and reports per-camera frame rate — this is how the Phase 1 gate ("clean 30-minute log with all camera streams") gets verified at the start of every drive.

**Files:**
- Create: `selfdrive/telecamerad/check_rig.py`
- Test: `selfdrive/telecamerad/tests/test_check_rig.py`

- [ ] **Step 1: Write the failing test (pure-logic evaluation function)**

```python
from openpilot.selfdrive.telecamerad.check_rig import evaluate


def test_evaluate_all_good():
  ok, report = evaluate({0: 200, 1: 195}, duration_s=10, expected_cams=2)
  assert ok
  assert "cam0" in report and "cam1" in report


def test_evaluate_one_camera_low():
  ok, report = evaluate({0: 200, 1: 30}, duration_s=10, expected_cams=2)
  assert not ok
  assert "LOW" in report


def test_evaluate_missing_camera():
  ok, _ = evaluate({0: 200}, duration_s=10, expected_cams=2)
  assert not ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest selfdrive/telecamerad/tests/test_check_rig.py -v`
Expected: FAIL with `ModuleNotFoundError` for `check_rig`.

- [ ] **Step 3: Write `selfdrive/telecamerad/check_rig.py`**

```python
#!/usr/bin/env python3
"""Telephoto rig bring-up check.

Watches telephotoCameraState and reports per-camera frame rate.
Usage (with telecamerad running): python -m selfdrive.telecamerad.check_rig [duration_s]
Exits 0 if every expected camera holds >= 70% of nominal FPS.
"""
import os
import sys
import time
from collections import Counter

from cereal import messaging

FPS = 20
RATE_TOLERANCE = 0.7


def evaluate(counts, duration_s, expected_cams, fps=FPS, tol=RATE_TOLERANCE):
  ok = True
  lines = []
  for idx in range(expected_cams):
    rate = counts.get(idx, 0) / duration_s
    good = rate >= tol * fps
    ok = ok and good
    lines.append(f"cam{idx}: {rate:5.1f} Hz {'OK' if good else 'LOW'}")
  return ok, "\n".join(lines)


def main(duration_s=10.0):
  expected_cams = sum(os.getenv(f"TELEPHOTO_CAM_{i}") is not None for i in (0, 1))
  if expected_cams == 0:
    print("no TELEPHOTO_CAM_* env vars set")
    return 1

  sock = messaging.sub_sock("telephotoCameraState", conflate=False)
  counts = Counter()
  end = time.monotonic() + duration_s
  while time.monotonic() < end:
    for msg in messaging.drain_sock(sock, wait_for_one=False):
      counts[msg.telephotoCameraState.cameraIndex] += 1
    time.sleep(0.05)

  ok, report = evaluate(counts, duration_s, expected_cams)
  print(report)
  return 0 if ok else 1


if __name__ == "__main__":
  duration = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
  sys.exit(main(duration))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest selfdrive/telecamerad/tests/test_check_rig.py -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add selfdrive/telecamerad/check_rig.py selfdrive/telecamerad/tests/test_check_rig.py
git commit -m "telecamerad: add rig bring-up check tool"
```

---

### Task 7: README + full verification

**Files:**
- Create: `selfdrive/telecamerad/README.md`

- [ ] **Step 1: Write the README**

````markdown
# telecamerad

Streams two USB telephoto cameras into openpilot for the stop-sign /
traffic-light stop policy project (see
`docs/superpowers/specs/2026-06-12-stop-sign-traffic-light-e2e-design.md`).

## Usage

```sh
export TELEPHOTO_CAM_0=/dev/video2        # required
export TELEPHOTO_CAM_1=/dev/video4        # optional second camera
export TELEPHOTO_WIDTH=1920               # optional, default 1920
export TELEPHOTO_HEIGHT=1080              # optional, default 1080
export TELEPHOTO_RECORD_DIR=/data/telephoto  # optional: record training footage
```

With `TELEPHOTO_CAM_0` set, the manager starts telecamerad onroad. It:

- publishes frames on the VisionIPC server `"telecamerad"` (stream slots
  ROAD/WIDE_ROAD = telephoto cam 0/1 — slots are reused, these are NOT the
  built-in cameras)
- publishes `telephotoCameraState` (logged in rlogs)
- if `TELEPHOTO_RECORD_DIR` is set, records H.264 mp4 segments with a
  `.jsonl` frameId sidecar for the training pipeline

## Rig check (Phase 1 gate)

With telecamerad running:

```sh
python -m selfdrive.telecamerad.check_rig 10
```

Exit 0 = every camera holding ≥70% of 20 FPS. The Phase 1 gate is a clean
30-minute onroad log with both telephoto cameras passing this check at start
and end of the drive.
````

- [ ] **Step 2: Run the full new test suite plus touched areas**

Run: `pytest selfdrive/telecamerad/ cereal/ system/manager/tests/ -q`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add selfdrive/telecamerad/README.md
git commit -m "telecamerad: add README with usage and Phase 1 gate procedure"
```

---

## Out of scope for this plan

- `stopPolicy` message, `stopmodeld`, planner changes, shadow mode → Plan B
- Auto-labeling / training / export → Plan C
- CARLA scenarios → Plan D
- On-device camera mounting/exposure tuning — physical work, no code

## Manual hardware validation (after all tasks, on the comma 4)

1. Plug in both cameras, `export TELEPHOTO_CAM_0=... TELEPHOTO_CAM_1=... TELEPHOTO_RECORD_DIR=/data/telephoto`, go onroad.
2. Run `python -m selfdrive.telecamerad.check_rig 10` → exit 0.
3. Drive 30 minutes; confirm rlog contains `telephotoCameraState` at ~40 Hz and mp4 segments exist with matching sidecars.
4. That log is the Phase 1 gate artifact.
