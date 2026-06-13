"""
Stop-policy training loop.

WindowDataset: contiguous sliding windows over precomputed episode features.
train(): episode-level 80/20 split, GRU policy training with BCEWithLogits +
         masked L1 losses, cosine LR schedule, best.pt / metrics.csv output.

Norm constants:
  dist_scale = 100.0  (metres — typical approach distance)
  speed_scale = 12.0  (m/s  — CRUISE speed from gen_metadrive.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tools.stoppolicy.data.episode import load_episode
from tools.stoppolicy.model import AuxPresenceHead, StopPolicyHead

# ---------------------------------------------------------------------------
# Norm constants
# ---------------------------------------------------------------------------
DIST_SCALE = 100.0   # metres
SPEED_SCALE = 12.0   # m/s (CRUISE)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
  """Sliding-window dataset over precomputed episode features.

  For each episode dir:
    - loads feats.npy  (N, 768) fp16 -> float32
    - loads labels via load_episode
    - builds all contiguous windows of length `window` (stride 1)

  __getitem__ returns:
    feats  : (window, 768) float32
    speeds : (window, 1)   float32
    targets: dict of (window,) float32 arrays:
               stop_prob, dist_to_stop, desired_speed, dist_mask

  oversample_stops=True: repeats windows that contain any stop_prob==1
  so they make up ~50 % of the final index list (seeded RNG).
  """

  def __init__(self, episode_dirs: list[str], window: int = 20,
               oversample_stops: bool = False, seed: int = 0):
    self.window = window
    self._windows: list[tuple[np.ndarray, np.ndarray, dict]] = []

    for ep_dir in episode_dirs:
      ep_dir = Path(ep_dir)
      feat_path = ep_dir / "feats.npy"
      if not feat_path.exists():
        continue
      feats = np.load(feat_path).astype(np.float32)  # (N, 768)
      ep = load_episode(ep_dir)
      labels = ep.labels
      N = min(len(feats), len(labels))
      if N < window:
        continue

      speeds = np.array([l["speed"] for l in labels[:N]], dtype=np.float32)
      stop_probs = np.array([l["stop_prob"] for l in labels[:N]], dtype=np.float32)
      dist_to_stops = np.array([l["dist_to_stop"] for l in labels[:N]], dtype=np.float32)
      desired_speeds = np.array([l["desired_speed"] for l in labels[:N]], dtype=np.float32)

      for start in range(N - window + 1):
        end = start + window
        w_feats = feats[start:end]
        w_speeds = speeds[start:end, None]   # (window, 1)
        w_stop = stop_probs[start:end]
        w_dist = dist_to_stops[start:end]
        w_dspd = desired_speeds[start:end]
        w_mask = (w_stop == 1.0).astype(np.float32)
        targets = {
          "stop_prob": w_stop,
          "dist_to_stop": w_dist,
          "desired_speed": w_dspd,
          "dist_mask": w_mask,
        }
        self._windows.append((w_feats, w_speeds, targets))

    # Build index list, optionally oversampling stop windows
    rng = random.Random(seed)
    stop_idxs = [i for i, (_, _, t) in enumerate(self._windows) if t["stop_prob"].max() == 1.0]
    nostop_idxs = [i for i, (_, _, t) in enumerate(self._windows) if t["stop_prob"].max() < 1.0]

    if oversample_stops and stop_idxs and nostop_idxs:
      # Repeat stop windows until they're ~50% of total
      n_total = len(self._windows)
      n_stop_needed = n_total  # so stop:nostop ≈ 1:1 → each 50%
      repeated = (stop_idxs * (n_stop_needed // len(stop_idxs) + 1))[:n_stop_needed]
      all_idxs = nostop_idxs + repeated
      rng.shuffle(all_idxs)
      self._index = all_idxs
    else:
      self._index = list(range(len(self._windows)))

  def __len__(self) -> int:
    return len(self._index)

  def __getitem__(self, i: int):
    feats, speeds, targets = self._windows[self._index[i]]
    return feats, speeds, {k: v for k, v in targets.items()}


# ---------------------------------------------------------------------------
# Online aux dataset (optional)
# ---------------------------------------------------------------------------

class OnlineDataset(Dataset):
  """Flat frame dataset for aux presence head. Loads feats.npy + labels.jsonl."""

  def __init__(self, online_root: Path):
    self._feats: list[np.ndarray] = []
    self._labels: list[np.ndarray] = []  # (2,) float32: has_light, has_sign

    for sub in sorted(online_root.iterdir()):
      feat_p = sub / "feats.npy"
      lbl_p = sub / "labels.jsonl"
      if not feat_p.exists() or not lbl_p.exists():
        continue
      feats = np.load(feat_p).astype(np.float32)
      lines = lbl_p.read_text().splitlines()
      for idx, line in enumerate(lines):
        if idx >= len(feats):
          break
        rec = json.loads(line)
        lbl = np.array([float(rec.get("has_light", 0)), float(rec.get("has_sign", 0))],
                       dtype=np.float32)
        self._feats.append(feats[idx])
        self._labels.append(lbl)

  def __len__(self) -> int:
    return len(self._feats)

  def __getitem__(self, i: int):
    return self._feats[i], self._labels[i]


# ---------------------------------------------------------------------------
# Helper: collate targets dict
# ---------------------------------------------------------------------------

def _collate(batch):
  feats = torch.from_numpy(np.stack([b[0] for b in batch]))
  speeds = torch.from_numpy(np.stack([b[1] for b in batch]))
  keys = batch[0][2].keys()
  targets = {k: torch.from_numpy(np.stack([b[2][k] for b in batch])) for k in keys}
  return feats, speeds, targets


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
  data_root,
  online_root,
  run_dir,
  epochs: int = 8,
  seed: int = 0,
  device: str = "cpu",
  lr: float = 3e-4,
  batch_size: int = 64,
  window: int = 20,
) -> dict:
  """Train StopPolicyHead; return final-epoch val metrics dict."""
  data_root = Path(data_root)
  run_dir = Path(run_dir)
  run_dir.mkdir(parents=True, exist_ok=True)

  # Deterministic seeding
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Discover episode dirs
  ep_dirs = sorted([d for d in data_root.iterdir()
                    if d.is_dir() and (d / "feats.npy").exists()])
  if len(ep_dirs) < 2:
    raise ValueError(f"Need ≥2 episode dirs with feats.npy, found {len(ep_dirs)}")

  # Episode-level split (~80/20, no window leakage)
  rng = random.Random(seed)
  shuffled = ep_dirs[:]
  rng.shuffle(shuffled)
  n_val = max(1, len(shuffled) // 5)
  n_train = max(1, len(shuffled) - n_val)
  train_eps = [str(e) for e in shuffled[:n_train]]
  val_eps = [str(e) for e in shuffled[n_train:]]

  # Save norm constants
  norm = {"dist_scale": DIST_SCALE, "speed_scale": SPEED_SCALE}
  (run_dir / "norm.json").write_text(json.dumps(norm))

  # Datasets
  train_ds = WindowDataset(train_eps, window=window, oversample_stops=True, seed=seed)
  val_ds = WindowDataset(val_eps, window=window, oversample_stops=False, seed=seed)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_collate, drop_last=False)
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          collate_fn=_collate, drop_last=False)

  # Model
  policy = StopPolicyHead().to(device)
  aux_head: Optional[AuxPresenceHead] = None
  online_loader = None

  if online_root is not None:
    online_root = Path(online_root)
    aux_head = AuxPresenceHead().to(device)
    online_ds = OnlineDataset(online_root)
    if len(online_ds) > 0:
      online_loader = DataLoader(online_ds, batch_size=batch_size, shuffle=True, drop_last=False)

  # Optimizer + cosine scheduler
  params = list(policy.parameters())
  if aux_head is not None:
    params += list(aux_head.parameters())
  optimizer = torch.optim.AdamW(params, lr=lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

  # Loss functions
  bce = nn.BCEWithLogitsLoss()

  # CSV writer
  csv_path = run_dir / "metrics.csv"
  csv_file = open(csv_path, "w", newline="")
  writer = csv.writer(csv_file)
  writer.writerow(["epoch", "train_loss", "stop_recall", "false_stop_rate", "dist_mae"])

  best_score = -float("inf")
  best_state = None
  final_metrics: dict = {}

  online_iter = iter(online_loader) if online_loader else None

  for epoch in range(1, epochs + 1):
    # ---- Train ----
    policy.train()
    if aux_head is not None:
      aux_head.train()
    total_loss = 0.0
    n_batches = 0

    for feats, speeds, targets in train_loader:
      feats = feats.to(device)           # (B, T, 768)
      speeds = speeds.to(device)         # (B, T, 1)
      stop_tgt = targets["stop_prob"].to(device)       # (B, T)
      dist_tgt = targets["dist_to_stop"].to(device)    # (B, T)
      dspd_tgt = targets["desired_speed"].to(device)   # (B, T)
      dist_mask = targets["dist_mask"].to(device)      # (B, T)

      optimizer.zero_grad()
      out = policy.forward_sequence(feats, speeds)
      # out shapes: stop_logit (B,T,1), dist_to_stop (B,T,1), desired_speed (B,T,1)
      stop_logit = out["stop_logit"].squeeze(-1)   # (B, T)
      dist_pred = out["dist_to_stop"].squeeze(-1)  # (B, T)
      dspd_pred = out["desired_speed"].squeeze(-1) # (B, T)

      # Stop loss (BCE)
      loss_stop = bce(stop_logit, stop_tgt)

      # Dist loss (masked L1, normalised)
      n_masked = dist_mask.sum()
      if n_masked > 0:
        loss_dist = (torch.abs(dist_pred / DIST_SCALE - dist_tgt / DIST_SCALE) * dist_mask).sum() / n_masked
      else:
        loss_dist = torch.tensor(0.0, device=device)

      # Speed loss (L1, normalised)
      loss_speed = torch.mean(torch.abs(dspd_pred / SPEED_SCALE - dspd_tgt / SPEED_SCALE))

      loss = loss_stop + 0.5 * loss_dist + 0.5 * loss_speed

      # Aux online loss
      if aux_head is not None and online_iter is not None:
        try:
          o_feats, o_lbls = next(online_iter)
        except StopIteration:
          online_iter = iter(online_loader)
          o_feats, o_lbls = next(online_iter)
        o_feats = o_feats.to(device)
        o_lbls = o_lbls.to(device)
        aux_logits = aux_head(o_feats)
        loss_aux = bce(aux_logits, o_lbls)
        loss = loss + 0.2 * loss_aux

      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      n_batches += 1

    scheduler.step()
    avg_loss = total_loss / max(n_batches, 1)

    # ---- Eval ----
    policy.eval()
    stop_recall_num = 0
    stop_recall_den = 0
    false_stop_num = 0
    false_stop_den = 0
    dist_mae_sum = 0.0
    dist_mae_n = 0

    with torch.no_grad():
      for feats, speeds, targets in val_loader:
        feats = feats.to(device)
        speeds = speeds.to(device)
        stop_tgt = targets["stop_prob"].numpy()       # (B, T)
        dist_tgt = targets["dist_to_stop"].numpy()   # (B, T)

        out = policy.forward_sequence(feats, speeds)
        stop_prob_pred = torch.sigmoid(out["stop_logit"].squeeze(-1)).cpu().numpy()  # (B, T)
        dist_pred_np = out["dist_to_stop"].squeeze(-1).cpu().numpy()                 # (B, T)

        B = feats.shape[0]
        for b in range(B):
          has_stop = stop_tgt[b].max() == 1.0
          max_pred = stop_prob_pred[b].max()
          if has_stop:
            stop_recall_den += 1
            if max_pred > 0.5:
              stop_recall_num += 1
            # dist MAE over stop frames
            stop_frames = stop_tgt[b] == 1.0
            if stop_frames.any():
              dist_mae_sum += np.abs(dist_pred_np[b][stop_frames] - dist_tgt[b][stop_frames]).mean()
              dist_mae_n += 1
          else:
            false_stop_den += 1
            if max_pred > 0.5:
              false_stop_num += 1

    stop_recall = stop_recall_num / max(stop_recall_den, 1)
    false_stop_rate = false_stop_num / max(false_stop_den, 1)
    dist_mae = dist_mae_sum / max(dist_mae_n, 1)

    print(f"epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
          f"recall={stop_recall:.3f}  fsr={false_stop_rate:.3f}  mae={dist_mae:.1f}m")
    writer.writerow([epoch, f"{avg_loss:.6f}", f"{stop_recall:.4f}",
                     f"{false_stop_rate:.4f}", f"{dist_mae:.4f}"])
    csv_file.flush()

    # Save best checkpoint
    score = stop_recall - false_stop_rate
    if score > best_score:
      best_score = score
      best_state = {
        "policy": {k: v.cpu().clone() for k, v in policy.state_dict().items()},
        "aux": ({k: v.cpu().clone() for k, v in aux_head.state_dict().items()}
                if aux_head is not None else None),
      }
      torch.save(best_state, run_dir / "best.pt")

    final_metrics = {"stop_recall": stop_recall, "false_stop_rate": false_stop_rate,
                     "dist_mae": dist_mae}

  csv_file.close()
  return final_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
  parser = argparse.ArgumentParser(description="Train stop-policy head")
  parser.add_argument("--data-root", required=True)
  parser.add_argument("--online-root", default=None)
  parser.add_argument("--run-name", default="default")
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--window", type=int, default=20)
  args = parser.parse_args()

  run_dir = Path("runs") / args.run_name
  metrics = train(
    data_root=args.data_root,
    online_root=args.online_root,
    run_dir=run_dir,
    epochs=args.epochs,
    seed=args.seed,
    device=args.device,
    lr=args.lr,
    batch_size=args.batch_size,
    window=args.window,
  )
  print("Final metrics:", metrics)


if __name__ == "__main__":
  _main()
