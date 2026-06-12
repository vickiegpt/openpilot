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
