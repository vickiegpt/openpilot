import json

import numpy as np

from tools.stoppolicy.train import WindowDataset, train


def synth_episode(tmp_path, name, red):
  """20s episode: red ones decelerate mid-episode; feats are label-correlated so
  the model can actually learn — channel 0 carries the stop signal."""
  from tools.stoppolicy.data.episode import EpisodeWriter
  ep = tmp_path / name
  T = 200
  with EpisodeWriter(ep, source="synth", fps=10) as w:
    for i in range(T):
      stopping = red and 50 <= i < 150
      w.add_frame(np.zeros((28, 28, 3), dtype=np.uint8), speed=12.0 if not stopping else 2.0,
                  stop_prob=1.0 if stopping else 0.0,
                  dist_to_stop=max(0.0, 100.0 - i) if stopping else -1.0,
                  desired_speed=2.0 if stopping else 12.0,
                  light_state="red" if stopping else "none")
  feats = np.zeros((T, 768), dtype=np.float16)
  for i in range(T):
    feats[i, 0] = 1.0 if (red and 50 <= i < 150) else -1.0
  feats += np.random.randn(T, 768).astype(np.float16) * 0.01
  np.save(ep / "feats.npy", feats)
  return ep


def test_window_dataset_split_and_oversample(tmp_path):
  eps = [synth_episode(tmp_path, f"ep{i}", red=i % 2 == 0) for i in range(6)]
  ds = WindowDataset([str(e) for e in eps], window=20, oversample_stops=True, seed=0)
  feats, speeds, targets = ds[0]
  assert feats.shape == (20, 768)
  assert speeds.shape == (20, 1)
  assert targets["stop_prob"].shape == (20,)
  frac_stop = np.mean([ds[i][2]["stop_prob"].max() for i in range(len(ds))])
  assert frac_stop > 0.3  # oversampled


def test_train_learns_synthetic(tmp_path):
  for i in range(8):
    synth_episode(tmp_path / "data", f"ep{i}", red=i % 2 == 0)
  metrics = train(data_root=tmp_path / "data", online_root=None, run_dir=tmp_path / "run",
                  epochs=8, seed=0, device="cpu")
  assert metrics["stop_recall"] > 0.8
  assert metrics["false_stop_rate"] < 0.2
  assert (tmp_path / "run" / "best.pt").exists()
  assert (tmp_path / "run" / "metrics.csv").exists()
  norm = json.loads((tmp_path / "run" / "norm.json").read_text())
  assert "dist_scale" in norm and "speed_scale" in norm
