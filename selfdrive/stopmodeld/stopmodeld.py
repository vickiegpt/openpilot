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
    if isinstance(outs, dict):
      outs = tuple(outs.values())
    return tuple(np.asarray(o.numpy(), np.float32) for o in outs)
  return runner


def main():
  if not Params().get_bool("StopPolicyEnabled"):
    cloudlog.warning("stopmodeld: StopPolicyEnabled not set, exiting")
    return

  from msgq.visionipc import VisionIpcClient, VisionStreamType
  import time
  pkl_path = os.path.join(os.path.dirname(__file__), "models", "stop_device.pkl")
  model = StopModeld(runner=_tinygrad_runner(pkl_path))

  vipc = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while not vipc.connect(False):
    time.sleep(0.1)
  assert vipc.is_connected()
  cloudlog.warning(f"stopmodeld: connected with buffer size: {vipc.buffer_len}")

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
