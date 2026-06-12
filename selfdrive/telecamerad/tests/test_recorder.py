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
