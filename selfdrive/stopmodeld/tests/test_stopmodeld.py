import numpy as np

from openpilot.selfdrive.stopmodeld.stopmodeld import StopDecider, preprocess_nv12, StopModeld
from tools.stoppolicy.model import HIDDEN_DIM


def test_hysteresis():
  d = StopDecider(enter_thresh=0.6, exit_thresh=0.4)
  assert d.update(0.5) is False
  assert d.update(0.7) is True
  assert d.update(0.5) is True
  assert d.update(0.3) is False


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
    return (np.array([[5.0]], np.float32), np.array([[8.0]], np.float32),
            np.array([[2.0]], np.float32), (hidden + 1.0).astype(np.float32))
  m = StopModeld(runner=runner)
  img = np.zeros((1, 3, 252, 448), np.float32)
  f1 = m.step(img, speed=10.0)
  assert f1["shouldStop"] is True
  assert f1["stopProb"] > 0.99
  assert abs(f1["desiredSpeed"] - 2.0) < 1e-5
  assert abs(f1["distToStop"] - 8.0) < 1e-5
  assert np.allclose(seen["hidden_in"], 0.0)
  m.step(img, speed=10.0)
  assert np.allclose(seen["hidden_in"], 1.0)


def test_desired_speed_never_negative():
  def runner(image, speed, hidden):
    return (np.array([[-5.0]], np.float32), np.array([[-1.0]], np.float32),
            np.array([[-3.0]], np.float32), np.zeros((1, HIDDEN_DIM), np.float32))
  m = StopModeld(runner=runner)
  f = m.step(np.zeros((1, 3, 252, 448), np.float32), speed=0.0)
  assert f["desiredSpeed"] >= 0.0
  assert f["shouldStop"] is False
