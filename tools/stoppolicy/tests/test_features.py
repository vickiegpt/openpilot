import numpy as np
import pytest

from tools.stoppolicy.features import FeatureExtractor, precompute_dir


@pytest.fixture(scope="module")
def fx():
  return FeatureExtractor(device="cuda")


@pytest.mark.slow
def test_extractor_shape(fx):
  imgs = np.random.randint(0, 255, (3, 252, 448, 3), dtype=np.uint8)
  f = fx.encode(imgs)
  assert f.shape == (3, 768)
  assert f.dtype == np.float16


@pytest.mark.slow
def test_precompute_episode(fx, tmp_path):
  from tools.stoppolicy.data.episode import EpisodeWriter
  with EpisodeWriter(tmp_path / "ep", source="test", fps=10) as w:
    for i in range(4):
      w.add_frame(np.random.randint(0, 255, (252, 448, 3), dtype=np.uint8),
                  speed=0.0, stop_prob=0.0, dist_to_stop=-1.0, desired_speed=0.0, light_state="none")
  precompute_dir(tmp_path / "ep", fx)
  f = np.load(tmp_path / "ep" / "feats.npy")
  assert f.shape == (4, 768)
