# Stop Sign & Traffic Light End-to-End Stop Policy — Design

**Date:** 2026-06-12
**Status:** Approved for planning

## Goal

Train and deploy a model that makes openpilot stop for stop signs and red
traffic lights (and proceed on green), at "general daily-driving aid" quality:
works on roads it has not seen, smooth comfortable stops, driver supervises at
all times.

## Context & Constraints

- **Target hardware:** comma 4 running openpilot, with an eGPU via the new
  usbgpu path for extra inference compute.
- **Extra sensors:** two USB telephoto (narrow-FOV) cameras aimed forward to
  see lights and signs far beyond the built-in road/wide cameras. They augment,
  not replace, the built-in cameras.
- **Training compute:** one consumer GPU (RTX 3090/4090 class).
- **Existing repo facts:** openpilot has no stop sign / traffic light handling
  today. Cameras publish via VisionIPC; `tools/webcam/camerad.py` is the
  pattern for external cameras. Models run via tinygrad from compiled ONNX
  (`selfdrive/modeld/modeld.py`, `dmonitoringmodeld.py`). Longitudinal
  planning lives in `selfdrive/controls/plannerd.py` /
  `selfdrive/controls/lib/longitudinal_planner.py`. No training tooling
  exists in-repo; training happens externally.

## Chosen Approach: Distilled End-to-End Policy

A frozen pretrained vision backbone (DINOv2) encodes frames from the two
telephoto cameras plus the road camera. A small temporal head (GRU or compact
transformer over ~2 s of features, a few million parameters) is trained
end-to-end to output the stopping decision. Only the head is trained.

Rationale: pure pixels-to-policy training requires fleet-scale data and would
overfit self-collected routes; a detector-plus-rules design was explicitly not
wanted. The frozen-backbone approach keeps the *decision* end-to-end-learned
while remaining trainable on one GPU, and the pretrained features generalize
to unseen roads and shrink the sim-to-real gap.

### Rejected alternatives

- **Pure pixels-to-policy (comma-style):** needs millions of miles; with
  self-collected data it memorizes known routes and misses the quality bar.
- **Detector + learned timing head:** most debuggable, least data-hungry, but
  perception is no longer learned end-to-end; user declined this direction.

## On-Car Architecture

```
2x USB telephoto cams        built-in road cam
        │                          │
   telecamerad ──VisionIPC──┐      │ (existing camerad)
                            ▼      ▼
                      stopmodeld (eGPU, tinygrad)
                  frozen DINOv2 backbone → temporal head
                            │
                     stopPolicy (cereal msg)
                            │
              longitudinal_planner.py (safety-bounded)
                            │
                      smooth stop / resume
```

### New components

1. **`telecamerad`** — daemon modeled on `tools/webcam/camerad.py`. Captures
   the two USB telephoto cameras and publishes them as new VisionIPC streams.
   Frames are logged by loggerd like any other camera, so every drive
   produces training data.

2. **`stopmodeld`** — model runner modeled on
   `selfdrive/modeld/dmonitoringmodeld.py`, running on the eGPU. At ~10 Hz it
   encodes the two telephoto frames + road frame with the frozen backbone,
   maintains ~2 s of feature history in the temporal head, and publishes:

   ```
   stopPolicy {
     stopProb        # probability we should be stopping for something ahead
     distToStopLine  # meters to where the car should be stationary
     desiredSpeed    # policy speed target
     modelValid      # frames fresh, eGPU healthy
   }
   ```

3. **Planner integration** — `longitudinal_planner.py` consumes `stopPolicy`
   and converts it into a stop target, bounded by hard-coded safety rules
   (see Safety Invariants). With no/stale `stopPolicy`, behavior is exactly
   stock — the integration must be a strict no-op when the feature is absent.

4. **Shadow mode** — a mode where `stopmodeld` runs and logs predictions but
   the planner ignores them. Used for weeks of validation: compare "what the
   model would have done" against actual human driving.

## Data Pipeline & Training

### Auto-labeling (no manual annotation)

An offline pipeline on the training PC:

1. Run a heavyweight open-vocabulary detector (Grounding DINO / YOLO-World)
   over logged telephoto frames to find traffic lights (+ color) and stop
   signs. This detector is a labeling tool only; it is never deployed.
2. Cross-reference ego behavior from logs: decelerating to standstill near a
   detected red/stop sign = positive stop event; the actual stopping point
   (from odometry) is the ground-truth stop line. Passing a green at speed =
   negative.
3. Targets fall out of the logs: `distToStopLine` from integrated odometry to
   the eventual stop point; `desiredSpeed` from the future human speed
   profile.

### Data sources and roles

- **Own drives (core):** every shadow-mode drive adds labeled stop/go events
  on the exact deployed camera geometry.
- **Public datasets (nuScenes, BDD100K):** auxiliary supervision — a side
  head predicts light/sign presence and state from backbone features, keeping
  features attentive to small distant lights and reducing overfitting.
- **CARLA sim:** scripted rare cases — occluded lights, yellow-light
  dilemmas, night, rain, flashing reds. The shared pretrained backbone keeps
  the sim-to-real gap at the feature level small.

### Training recipe (single consumer GPU)

- Precompute and cache DINOv2 features for all logged frames once; train the
  head against cached features (epochs in minutes).
- Head: GRU or small transformer, ~2 s temporal window, 3 camera streams,
  a few million trainable parameters.
- Losses: BCE on `stopProb`; regression on `distToStopLine` (only during stop
  events) and `desiredSpeed`; auxiliary detection loss on public data.
- Oversample stop events to counter the ~95 % "don't stop" class imbalance.
- Export: PyTorch → ONNX → tinygrad compile, same pipeline as `modeld`.

### Evaluation

- **Geographic split:** held-out routes never used in training (a random
  frame split would leak).
- Metrics: stop recall (safety), false stops per 100 km (comfort),
  stop-point error in meters, red-vs-green discrimination.
- Closed-loop CARLA scenario suite run on every checkpoint before it goes in
  the car.

## Safety Invariants (hard-coded, never learned)

- Policy can only request deceleration; it can never raise the speed target.
- Deceleration capped at comfortable limits; emergencies remain driver/AEB.
- Hysteresis on `stopProb` prevents brake oscillation from borderline
  predictions.
- Stale frames, eGPU dropout, or `modelValid == false` → instant silent
  fallback to stock behavior.
- Driver gas input overrides any model stop; overrides are logged as
  high-value training signal.
- UI indicates why the car is slowing ("stopping for light/sign ahead") so
  supervision is meaningful.

## Rollout Phases (each with a gate)

1. **Rig bring-up** — telephoto cameras streaming and logging.
   *Gate: clean 30-minute log with all five camera streams.*
2. **Data + first model** — collect, auto-label, train, evaluate offline.
   *Gate: held-out routes show >95 % stop recall and <1 false stop / 100 km.*
3. **Shadow mode** — on-car inference, zero control authority.
   *Gate: shadow metrics match offline metrics.*
4. **Supervised engagement on familiar routes.**
   *Gate: 25 consecutive clean stop events with zero uncommanded stops.*
5. **General supervised use** — the v1 goal.

## Testing

- Unit tests for planner integration with synthetic `stopPolicy` messages:
  stale, flickering, conflicting, absent.
- Process replay on logged segments proving the planner change is a no-op
  when `stopPolicy` is absent.
- CARLA closed-loop scenarios (red, green, late yellow, occluded light, stop
  sign) for every trained checkpoint.

## Non-Goals (v1)

- Unsupervised operation of any kind.
- Handling of intersections beyond stop/go at the policy's own stop line
  (no turn negotiation, no cross-traffic reasoning).
- Flashing-yellow / pedestrian-signal semantics beyond what sim scenarios
  cover.
- Upstreaming to comma.ai's openpilot.

## Note on Responsibility

An automated red-light/stop-sign responder is safety-critical. This design
keeps the driver as the supervising authority in every phase; phase gates are
to be treated as real release criteria, not suggestions.
