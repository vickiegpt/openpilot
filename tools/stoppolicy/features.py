"""DINOv2 ViT-S/14 feature extractor and directory-level precompute.

Per-dir feats.npy: shape (N, 768) fp16
  = concat of x_norm_clstoken (384) + mean(x_norm_patchtokens, dim=1) (384)
  at canonical input 448x252, ImageNet-normalized.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ImageNet stats
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

# Canonical resolution: H and W must be divisible by 14
CANONICAL_W = 448
CANONICAL_H = 252

BATCH_SIZE = 64


class FeatureExtractor:
  """Wraps DINOv2 ViT-S/14 for batch feature extraction."""

  def __init__(self, device="cuda"):
    self.device = torch.device(device)
    self.model = torch.hub.load(
      "facebookresearch/dinov2",
      "dinov2_vits14",
      skip_validation=True,
    )
    self.model.eval().to(self.device)

  def _preprocess(self, imgs: np.ndarray) -> torch.Tensor:
    """Convert NHWC uint8 numpy -> NCHW float32 tensor, resized + normalized."""
    N, H, W, C = imgs.shape
    # Convert to float tensor NCHW in [0, 1]
    t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255.0)
    # Resize if needed
    if H != CANONICAL_H or W != CANONICAL_W:
      t = F.interpolate(t, size=(CANONICAL_H, CANONICAL_W), mode="bilinear", align_corners=False)
    # ImageNet normalization
    mean = _MEAN.view(1, 3, 1, 1)
    std = _STD.view(1, 3, 1, 1)
    t = (t - mean) / std
    return t.to(self.device)

  def encode(self, imgs: np.ndarray) -> np.ndarray:
    """Encode NHWC uint8 images -> (N, 768) float16 numpy array.

    Runs in batches of BATCH_SIZE with torch.inference_mode.
    """
    N = imgs.shape[0]
    all_feats = []

    with torch.inference_mode():
      for start in range(0, N, BATCH_SIZE):
        batch_np = imgs[start:start + BATCH_SIZE]
        batch = self._preprocess(batch_np)
        out = self.model.forward_features(batch)
        cls_tok = out["x_norm_clstoken"]          # (B, 384)
        patch_tok = out["x_norm_patchtokens"]      # (B, num_patches, 384)
        patch_mean = patch_tok.mean(dim=1)          # (B, 384)
        feats = torch.cat([cls_tok, patch_mean], dim=1)  # (B, 768)
        all_feats.append(feats.cpu().to(torch.float16))

    return torch.cat(all_feats, dim=0).numpy()


def _load_images_ordered(img_dir: Path) -> np.ndarray:
  """Load all JPGs from a directory in sorted index order -> NHWC uint8.

  Images are resized to the canonical (CANONICAL_W x CANONICAL_H) at load time so
  they can be stacked: online data is stored at longest-side-448 with preserved
  aspect ratio, so frames vary in shape. encode() would resize anyway, so this is
  equivalent and only makes the stack possible.
  """
  paths = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem))
  frames = []
  for p in paths:
    img = Image.open(p).convert("RGB")
    if img.size != (CANONICAL_W, CANONICAL_H):
      img = img.resize((CANONICAL_W, CANONICAL_H), Image.BILINEAR)
    frames.append(np.asarray(img))
  return np.stack(frames)


def precompute_dir(path: Path, fx: FeatureExtractor) -> None:
  """Compute and save feats.npy for a single episode or online data dir.

  Episode dirs contain frames/, online dirs contain images/.
  Asserts that the image count matches the metadata count.
  """
  path = Path(path)
  frames_dir = path / "frames"
  images_dir = path / "images"

  if frames_dir.is_dir():
    # Episode directory
    meta = json.loads((path / "meta.json").read_text())
    expected = meta["n_frames"]
    imgs = _load_images_ordered(frames_dir)
  elif images_dir.is_dir():
    # Online data directory
    meta = json.loads((path / "meta.json").read_text())
    expected = meta["n_images"]
    imgs = _load_images_ordered(images_dir)
  else:
    raise ValueError(f"{path} is neither an episode dir (frames/) nor an online dir (images/)")

  assert len(imgs) == expected, (
    f"Image count mismatch in {path}: found {len(imgs)}, meta says {expected}"
  )

  feats = fx.encode(imgs)
  np.save(path / "feats.npy", feats.astype(np.float16))


def _walk_data_root(root: Path):
  """Recursively yield all dirs that contain frames/ or images/."""
  for child in sorted(root.rglob("*")):
    if child.is_dir():
      if (child / "frames").is_dir() or (child / "images").is_dir():
        yield child


def main():
  parser = argparse.ArgumentParser(description="Precompute DINOv2 features for all data dirs.")
  parser.add_argument("--data-root", required=True, type=Path,
                      help="Root dir; all immediate subdir trees are walked for episode/online dirs.")
  parser.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
  parser.add_argument("--force", action="store_true",
                      help="Re-compute even if feats.npy already exists.")
  args = parser.parse_args()

  print(f"Loading FeatureExtractor on {args.device}...")
  fx = FeatureExtractor(device=args.device)
  print("Model loaded.")

  dirs = list(_walk_data_root(args.data_root))
  print(f"Found {len(dirs)} data dirs under {args.data_root}")

  done = 0
  skipped = 0
  for d in dirs:
    feats_path = d / "feats.npy"
    if feats_path.exists() and not args.force:
      print(f"  SKIP {d}  (feats.npy exists)")
      skipped += 1
      continue
    print(f"  Computing {d} ...", end=" ", flush=True)
    precompute_dir(d, fx)
    n = np.load(feats_path).shape[0]
    print(f"done ({n} frames)")
    done += 1

  print(f"\nFinished: {done} computed, {skipped} skipped.")


if __name__ == "__main__":
  main()
