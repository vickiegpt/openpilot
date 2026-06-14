# Stop Policy (stop sign / traffic light) — training pipeline

A distilled end-to-end stop policy: a frozen DINOv2 backbone encodes camera
frames, and a small step-wise GRU head learns *when and where to stop* for
traffic lights and stop signs. Trains on one GPU from simulator + public
("online") data — no self-collected drives required for v1.

See the design spec:
`docs/superpowers/specs/2026-06-12-stop-sign-traffic-light-e2e-design.md`
and the plan: `docs/superpowers/plans/2026-06-12-stop-policy-training-v1.md`.

## Setup

```sh
bash tools/stoppolicy/setup_env.sh   # builds tools/stoppolicy/.venv
```

Reuses the system torch (CUDA) via `--system-site-packages`; installs
MetaDrive, HF `datasets`, onnx/onnxruntime. DINOv2 is fetched via `torch.hub`
on first use.

## Pipeline

Run from the repo root with the training venv
(`tools/stoppolicy/.venv/bin/python`):

```sh
PY=tools/stoppolicy/.venv/bin/python
DATA=/tmp/stoppolicy_data   # tmpfs: data is large, the root disk is small

# 1. Simulator episodes (scripted traffic-light approaches, exact labels)
$PY -m tools.stoppolicy.data.gen_metadrive --out $DATA/sim --n 60 --seed 1 --max-gb 1.0

# 2. Online detection data (COCO traffic-light/stop-sign presence, aux loss)
$PY -m tools.stoppolicy.data.fetch_online --out $DATA/online --n 2000 --max-gb 0.5

# 3. Precompute frozen DINOv2 features for every data dir
$PY -m tools.stoppolicy.features --data-root $DATA

# 4. Train the policy head (+ aux presence head on online data)
$PY -m tools.stoppolicy.train --data-root $DATA/sim --online-root $DATA/online \
    --run-name v1 --epochs 40 --seed 0

# 5. Export the streaming single-step model to ONNX
$PY -m tools.stoppolicy.export --checkpoint runs/v1/best.pt --out runs/v1/stop_policy.onnx
```

## Components

| File | Responsibility |
|------|----------------|
| `data/episode.py` | Episode storage: `frames/%06d.jpg` + `labels.jsonl` + `meta.json` |
| `data/gen_metadrive.py` | Scripted MetaDrive traffic-light episodes with ground-truth labels |
| `data/fetch_online.py` | Streams COCO presence-labeled images (`has_light`/`has_sign`) |
| `features.py` | DINOv2 ViT-S/14 → cached `feats.npy` (N, 768) fp16 per dir |
| `model.py` | `StopPolicyHead` (step-wise GRU) + `AuxPresenceHead` |
| `train.py` | Windowed training, stop-recall/false-stop/dist-MAE metrics |
| `export.py` | ONNX export of the single-step streaming model |

The model outputs, per frame: `stop_logit` (sigmoid → stop probability),
`dist_to_stop` (meters), `desired_speed` (m/s), and the recurrent
`hidden_out`. Normalization constants are saved alongside the checkpoint in
`norm.json`.

## v1 results (sim + online)

Trained on 60 MetaDrive episodes (12,618 frames; episode-level 80/20 split)
with 2000 COCO images as auxiliary presence supervision; 40 epochs, DINOv2
ViT-S/14 frozen backbone, ~1M-param GRU head, single GPU. Metrics are for the
best checkpoint by `stop_recall - false_stop_rate` on held-out sim episodes
(epoch 29 in the 2026-06-14 run).

| Metric | v1 | Bar |
|--------|----|-----|
| Stop recall | **1.000** | ≥ 0.90 |
| False stops / window | **0.073** | ≤ 0.10 |
| Stop-point dist MAE | **1.60 m** | ≤ 10 m |

All v1 criteria met. Artifacts (gitignored): `runs/v1/best.pt`,
`runs/v1/stop_policy.onnx`, `runs/v1/metrics.csv`, `runs/v1/norm.json`.
The fresh data run used `281M` in `/tmp/stoppolicy_data`; exported run
artifacts use `7.9M` in `runs/v1`.

### Caveats / next steps

- **Sim-only validation.** Metrics are on held-out *simulator* episodes; the
  online data only supervises object *presence*, not stop timing. Real-world
  generalization is unproven until the telephoto-rig drive data exists (Plan A)
  and real-drive auto-labeling is added.
- MetaDrive's traffic light is small at distance (4–8 px at 40–60 m); the
  signal is real but a future telephoto camera is what makes distant lights
  legible — that is the whole point of the two-camera rig.
- On-device integration (`stopmodeld` + longitudinal planner, Plan B) is not
  part of this pipeline; the ONNX export is streaming-ready (explicit hidden
  state) for that step.
