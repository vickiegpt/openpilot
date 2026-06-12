import time

import numpy as np
import pytest

from cereal import messaging
from msgq.visionipc import VisionIpcClient
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
  # TeleCamerad (and its PubMaster) must exist before sub_sock connects.
  # msgq's init_publisher resets num_readers=0, evicting any subscriber that
  # registered before the publisher initialized. Creating TeleCamerad first
  # then sleeping 0.2 s lets the subscriber join an already-live publisher.
  tcd = TeleCamerad(cameras=[FakeCamera()])
  sock = messaging.sub_sock('telephotoCameraState', conflate=False, timeout=100)
  time.sleep(0.2)  # let the subscriber connect to the live publisher
  tcd.camera_runner(0, tcd.cameras[0])
  msgs = drain(sock, N_FRAMES)
  assert len(msgs) == N_FRAMES
  assert [m.telephotoCameraState.frameId for m in msgs] == list(range(N_FRAMES))
  assert all(m.telephotoCameraState.cameraIndex == 0 for m in msgs)


def test_two_cameras_run_to_completion():
  # Same ordering requirement: TeleCamerad before sub_sock.
  tcd = TeleCamerad(cameras=[FakeCamera(), FakeCamera()])
  sock = messaging.sub_sock('telephotoCameraState', conflate=False, timeout=100)
  time.sleep(0.2)
  tcd.run()  # camera generators are finite, so this returns
  msgs = drain(sock, 2 * N_FRAMES)
  by_cam = {0: [], 1: []}
  for m in msgs:
    by_cam[m.telephotoCameraState.cameraIndex].append(m.telephotoCameraState.frameId)
  assert by_cam[0] == list(range(N_FRAMES))
  assert by_cam[1] == list(range(N_FRAMES))


def test_env_cameras_requires_cam0_for_cam1(monkeypatch):
  """TELEPHOTO_CAM_1 without TELEPHOTO_CAM_0 must raise SystemExit immediately."""
  monkeypatch.delenv("TELEPHOTO_CAM_0", raising=False)
  monkeypatch.setenv("TELEPHOTO_CAM_1", "/dev/video1")
  with pytest.raises(SystemExit, match="TELEPHOTO_CAM_1 requires TELEPHOTO_CAM_0"):
    from openpilot.selfdrive.telecamerad.telecamerad import env_cameras
    env_cameras()


def test_vipc_client_receives_frame():
  """A VisionIpcClient connected to the telecamerad server must receive at least one frame."""
  tcd = TeleCamerad(cameras=[FakeCamera()])
  # Create client before running; connect after server is up (it already is from __init__)
  client = VisionIpcClient("telecamerad", STREAM_SLOTS[0], False)
  assert client.connect(True)

  # Run one camera iteration on the calling thread so we stay single-threaded
  tcd.camera_runner(0, tcd.cameras[0])

  # recv with timeout — should have frames queued
  buf = client.recv(timeout_ms=500)
  assert buf is not None, "VisionIpcClient did not receive a frame within timeout"
  assert client.frame_id == 0  # first frame sent


def test_recording_enabled(tmp_path, monkeypatch):
  monkeypatch.setenv("TELEPHOTO_RECORD_DIR", str(tmp_path))
  tcd = TeleCamerad(cameras=[FakeCamera()])
  tcd.run()
  assert (tmp_path / "cam0_seg0000.mp4").exists()
  sidecar = (tmp_path / "cam0_seg0000.jsonl").read_text().splitlines()
  assert len(sidecar) == N_FRAMES


def test_recorder_failure_does_not_kill_publishing(tmp_path, monkeypatch):
  monkeypatch.setenv("TELEPHOTO_RECORD_DIR", str(tmp_path))
  tcd = TeleCamerad(cameras=[FakeCamera()])
  def boom(*a, **k):
    raise OSError("disk full")
  tcd.recorders[0].write = boom
  sock = messaging.sub_sock('telephotoCameraState', conflate=False, timeout=100)
  time.sleep(0.2)
  tcd.run()  # must not raise
  msgs = drain(sock, N_FRAMES)
  assert len(msgs) == N_FRAMES   # publishing survived
  assert 0 not in tcd.recorders  # recorder disabled
