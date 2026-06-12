import numpy as np

from tools.stoppolicy.data.episode import EpisodeWriter, load_episode


def make_frame(val):
  return np.full((96, 160, 3), val, dtype=np.uint8)


def test_episode_roundtrip(tmp_path):
  ep_dir = tmp_path / "ep0000"
  with EpisodeWriter(ep_dir, source="test", fps=10) as w:
    for i in range(5):
      w.add_frame(make_frame(i * 20), speed=float(i), stop_prob=float(i % 2),
                  dist_to_stop=10.0 * i if i % 2 else -1.0,
                  desired_speed=float(i) + 0.5, light_state="red" if i % 2 else "none")

  ep = load_episode(ep_dir)
  assert ep.meta["source"] == "test"
  assert ep.meta["fps"] == 10
  assert ep.meta["n_frames"] == 5
  assert len(ep.labels) == 5
  assert ep.labels[3]["stop_prob"] == 1.0
  assert ep.labels[3]["dist_to_stop"] == 30.0
  assert ep.labels[3]["light_state"] == "red"
  assert ep.labels[3]["t"] == 0.3
  frames = ep.load_frames()
  assert frames.shape == (5, 96, 160, 3)
  assert frames[2].mean() > frames[0].mean()  # content survived JPEG


def test_writer_enforces_byte_budget(tmp_path):
  with EpisodeWriter(tmp_path / "ep0001", source="test", fps=10, max_bytes=10_000) as w:
    wrote = 0
    for i in range(1000):
      if not w.add_frame(make_frame(i % 255), speed=0.0, stop_prob=0.0,
                         dist_to_stop=-1.0, desired_speed=0.0, light_state="none"):
        break
      wrote += 1
  assert 0 < wrote < 1000  # budget cut it off
