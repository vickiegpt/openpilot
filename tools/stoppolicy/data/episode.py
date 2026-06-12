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
