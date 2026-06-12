"""
Fetch COCO detection images from Hugging Face for auxiliary loss training.

Output contract (consumed by Tasks 4/6):
  out_dir/images/%06d.jpg
  out_dir/labels.jsonl   -- one JSON line per image: {idx, has_light, has_sign}
  out_dir/meta.json      -- {n_images, source}
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def presence_from_categories(cat_names: list[str]) -> tuple[bool, bool]:
  """Return (has_light, has_sign) from a list of category name strings.

  Case-insensitive match on 'traffic light' and 'stop sign'.
  """
  lower = [c.lower() for c in cat_names]
  has_light = any("traffic light" in c for c in lower)
  has_sign  = any("stop sign" in c for c in lower)
  return has_light, has_sign


# ---------------------------------------------------------------------------
# BudgetedSaver
# ---------------------------------------------------------------------------

class BudgetedSaver:
  """Save RGB images + labels until a byte budget is exhausted.

  Mirrors EpisodeWriter's budget pattern from episode.py.
  """

  def __init__(self, out_dir, max_bytes: int):
    self.out_dir = Path(out_dir)
    (self.out_dir / "images").mkdir(parents=True, exist_ok=True)
    self.max_bytes = max_bytes
    self.bytes_written = 0
    self.n = 0
    self.source = "unknown"
    self._labels_f = open(self.out_dir / "labels.jsonl", "w")
    self._closed = False

  def add(self, rgb: np.ndarray, has_light: bool, has_sign: bool) -> bool:
    """Save one image and its label. Returns False once budget is exceeded."""
    if self._closed or self.bytes_written >= self.max_bytes:
      return False
    p = self.out_dir / "images" / f"{self.n:06d}.jpg"
    Image.fromarray(rgb).save(p, quality=80)
    self.bytes_written += p.stat().st_size
    rec = {"idx": self.n, "has_light": has_light, "has_sign": has_sign}
    self._labels_f.write(json.dumps(rec) + "\n")
    self._labels_f.flush()
    self.n += 1
    return True

  def close(self):
    """Idempotent: write meta.json and close label file."""
    if self._closed:
      return
    self._closed = True
    self._labels_f.close()
    meta = {"n_images": self.n, "source": self.source}
    (self.out_dir / "meta.json").write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _resize_longest(img: Image.Image, longest: int = 448) -> np.ndarray:
  """Resize so the longest side == longest, keep aspect ratio, return RGB uint8."""
  w, h = img.size
  scale = longest / max(w, h)
  nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
  img = img.resize((nw, nh), Image.BILINEAR)
  return np.asarray(img.convert("RGB"))


def _load_streaming_dataset(candidate_ids: list[str]):
  """Try each dataset id in order; return (dataset, dataset_id) or raise."""
  import datasets as hf_datasets  # noqa: PLC0415 (import inside function for lazy load)

  for ds_id in candidate_ids:
    try:
      ds = hf_datasets.load_dataset(ds_id, split="train", streaming=True)
      return ds, ds_id
    except Exception as e:  # noqa: BLE001
      print(f"[fetch_online] {ds_id} unavailable: {e}")
  raise RuntimeError(f"None of the candidate datasets could be loaded: {candidate_ids}")


def _category_names_from_sample(sample: dict, features) -> list[str]:
  """Extract category name strings from a sample, handling ClassLabel or raw ints."""
  objs = sample.get("objects", {})
  cat_field = objs.get("category", [])

  # Try to get the ClassLabel feature so we can call int2str
  try:
    # features["objects"]["category"] is a Sequence(ClassLabel(...))
    cat_feature = features["objects"]["category"].feature
    if hasattr(cat_feature, "int2str"):
      return [cat_feature.int2str(c) for c in cat_field]
    if hasattr(cat_feature, "names"):
      names = cat_feature.names
      return [names[c] for c in cat_field]
  except Exception:  # noqa: BLE001
    pass

  # Fallback: already strings
  return [str(c) for c in cat_field]


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

CANDIDATE_DATASETS = [
  "detection-datasets/coco",
  "rafaelpadilla/coco2017",
]


def fetch(out_dir, n_target: int = 200, max_bytes: int = 500_000_000) -> int:
  """Stream COCO images from HF and save with presence labels.

  Args:
    out_dir:   destination directory (created if absent).
    n_target:  stop after saving this many images.
    max_bytes: hard byte budget (bytes of saved JPEGs).

  Returns:
    Number of images saved.
  """
  os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

  saver = BudgetedSaver(out_dir, max_bytes=max_bytes)
  ds, ds_id = _load_streaming_dataset(CANDIDATE_DATASETS)
  saver.source = ds_id
  features = ds.features

  print(f"[fetch_online] streaming {ds_id}")

  # 2:1 positive:negative ratio
  pos_count = 0
  neg_count = 0
  target_pos = (n_target * 2) // 3   # ~2/3 positives
  target_neg = n_target - target_pos  # ~1/3 negatives

  for sample in ds:
    if saver.n >= n_target:
      break

    cat_names = _category_names_from_sample(sample, features)
    has_light, has_sign = presence_from_categories(cat_names)
    is_positive = has_light or has_sign

    # Enforce 2:1 ratio: skip excess negatives
    if not is_positive:
      if neg_count >= target_neg and pos_count < target_pos:
        continue

    # Convert image
    img = sample["image"]
    if not isinstance(img, Image.Image):
      img = Image.fromarray(img)
    rgb = _resize_longest(img, longest=448)

    ok = saver.add(rgb, has_light=has_light, has_sign=has_sign)
    if not ok:
      break

    if is_positive:
      pos_count += 1
    else:
      neg_count += 1

  saver.close()
  print(f"[fetch_online] saved {saver.n} images ({pos_count} pos, {neg_count} neg) from {ds_id}")
  return saver.n


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
  import argparse

  parser = argparse.ArgumentParser(description="Fetch COCO detection images from HuggingFace")
  parser.add_argument("--out", required=True, help="Output directory")
  parser.add_argument("--n", type=int, default=200, help="Target number of images")
  parser.add_argument("--max-gb", type=float, default=0.5, help="Max disk budget in GB")
  args = parser.parse_args()

  max_bytes = int(args.max_gb * 1024 ** 3)
  n = fetch(out_dir=args.out, n_target=args.n, max_bytes=max_bytes)
  print(f"Done: {n} images written to {args.out}")


if __name__ == "__main__":
  _cli()
