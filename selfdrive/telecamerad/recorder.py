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
    self.closed = False

  def _open_segment(self):
    self._close_segment()
    self.seg_idx += 1
    base = self.out_dir / f"cam{self.camera_index}_seg{self.seg_idx:04d}"
    # fragmented mp4: footage stays decodable even if the daemon is killed mid-segment
    self.container = av.open(str(base.with_suffix(".mp4")), mode="w",
                             options={"movflags": "frag_keyframe+empty_moov"})
    self.stream = self.container.add_stream("h264", rate=self.fps,
                                            options={"preset": "veryfast", "tune": "zerolatency", "crf": "23"})
    self.stream.width, self.stream.height = self.w, self.h
    self.stream.pix_fmt = "nv12"
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
    if self.closed:
      return
    if self.container is None or self.frames_in_seg >= self.segment_len_frames:
      self._open_segment()
    arr = np.frombuffer(nv12_bytes, dtype=np.uint8).reshape(self.h * 3 // 2, self.w)
    frame = av.VideoFrame.from_ndarray(arr, format="nv12")
    frame.pts = self.pts
    self.pts += 1
    for pkt in self.stream.encode(frame):
      self.container.mux(pkt)
    self.sidecar.write(json.dumps({"frameId": frame_id, "timestampEof": timestamp_eof}) + "\n")
    self.sidecar.flush()
    self.frames_in_seg += 1

  def close(self):
    if self.closed:
      return
    self.closed = True
    self._close_segment()
