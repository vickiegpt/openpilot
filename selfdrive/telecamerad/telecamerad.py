#!/usr/bin/env python3
import os
import threading

from msgq.visionipc import VisionIpcServer, VisionStreamType
from cereal import messaging

from openpilot.common.realtime import Ratekeeper
from openpilot.tools.webcam.camera import Camera
from openpilot.selfdrive.telecamerad.recorder import TeleRecorder

# The telephoto cameras get their own VisionIPC server ("telecamerad").
# Stream types only need to be unique per server, so we reuse the ROAD and
# WIDE_ROAD slots instead of forking msgq to add new enum values. Clients
# must connect with the server name "telecamerad" — on this server these
# slots mean telephoto cam 0 and 1, not the built-in cameras.
STREAM_SLOTS = [VisionStreamType.VISION_STREAM_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD]
FPS = 20


def env_cameras():
  cam0_dev = os.getenv("TELEPHOTO_CAM_0")
  cam1_dev = os.getenv("TELEPHOTO_CAM_1")

  if cam1_dev is not None and cam0_dev is None:
    raise SystemExit("TELEPHOTO_CAM_1 requires TELEPHOTO_CAM_0")

  cams = []
  for idx, dev in enumerate([cam0_dev, cam1_dev]):
    if dev is not None:
      cams.append(Camera("telephotoCameraState", STREAM_SLOTS[idx], dev,
                         width=int(os.getenv("TELEPHOTO_WIDTH", "1920")),
                         height=int(os.getenv("TELEPHOTO_HEIGHT", "1080")),
                         fps=FPS, rotate_180=False))

  if len(cams) == 0:
    raise SystemExit("set TELEPHOTO_CAM_0 (and optionally TELEPHOTO_CAM_1)")

  return cams


class TeleCamerad:
  def __init__(self, cameras=None):
    self.cameras = cameras if cameras is not None else env_cameras()

    self.pm = messaging.PubMaster(["telephotoCameraState"])
    self.pm_lock = threading.Lock()

    self.vipc_server = VisionIpcServer("telecamerad")
    for idx, cam in enumerate(self.cameras):
      self.vipc_server.create_buffers(STREAM_SLOTS[idx], 20, int(cam.W), int(cam.H))
    self.vipc_server.start_listener()

    record_dir = os.getenv("TELEPHOTO_RECORD_DIR")
    self.recorders = {}
    if record_dir is not None:
      for idx, cam in enumerate(self.cameras):
        self.recorders[idx] = TeleRecorder(record_dir, idx, int(cam.W), int(cam.H), fps=FPS)

  def _send(self, idx, cam, yuv):
    eof = int(cam.cur_frame_id * (1 / FPS) * 1e9)
    self.vipc_server.send(STREAM_SLOTS[idx], yuv, cam.cur_frame_id, eof, eof)
    dat = messaging.new_message("telephotoCameraState", valid=True)
    dat.telephotoCameraState.frameId = cam.cur_frame_id
    dat.telephotoCameraState.cameraIndex = idx
    dat.telephotoCameraState.timestampEof = eof
    with self.pm_lock:
      self.pm.send("telephotoCameraState", dat)
    if idx in self.recorders:
      self.recorders[idx].write(yuv, cam.cur_frame_id, eof)

  def camera_runner(self, idx, cam):
    rk = Ratekeeper(FPS, None)
    for yuv in cam.read_frames():
      self._send(idx, cam, yuv)
      cam.cur_frame_id += 1
      rk.keep_time()

  def run(self):
    threads = [threading.Thread(target=self.camera_runner, args=(i, c)) for i, c in enumerate(self.cameras)]
    try:
      for t in threads:
        t.start()
      for t in threads:
        t.join()
    finally:
      for rec in self.recorders.values():
        rec.close()


def main():
  TeleCamerad().run()


if __name__ == "__main__":
  main()
