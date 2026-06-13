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


@pytest.mark.slow
def test_precompute_online_variable_shapes(fx, tmp_path):
  # Online images are stored at longest-side-448 with preserved aspect ratio, so
  # frames have varying shapes; precompute must handle (not np.stack-fail on) them.
  import json
  from PIL import Image
  d = tmp_path / "online"
  (d / "images").mkdir(parents=True)
  shapes = [(252, 448), (448, 336), (300, 448)]  # (H, W) — all different
  for i, (h, w) in enumerate(shapes):
    Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(d / "images" / f"{i:06d}.jpg")
  (d / "labels.jsonl").write_text(
    "".join(json.dumps({"idx": i, "has_light": bool(i % 2), "has_sign": False}) + "\n" for i in range(3)))
  (d / "meta.json").write_text(json.dumps({"n_images": 3, "source": "test"}))
  precompute_dir(d, fx)
  f = np.load(d / "feats.npy")
  assert f.shape == (3, 768)
