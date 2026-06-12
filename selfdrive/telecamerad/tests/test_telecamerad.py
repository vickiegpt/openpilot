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
