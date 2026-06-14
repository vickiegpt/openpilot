# stopmodeld — on-device stop policy

Runs the trained stop policy on the road camera and asks the longitudinal planner
to stop for lights/signs. Supervised, param-gated, decelerate-only.

Pipeline: `stopmodeld` (road cam → DINOv2+GRU tinygrad model) → `stopPolicy` msg →
`longitudinal_planner.py` (lowers cruise target / forces stop via the existing MPC
stop path). See `docs/superpowers/plans/2026-06-14-stop-policy-ondevice-plan-b.md`.

## Params
- `StopPolicyEnabled` (BOOL, persistent): runs the `stopmodeld` daemon. In shadow
  the planner only LOGS what it would do.
- `StopPolicyActive` (BOOL, persistent): grants the planner control authority
  (supervised). Only meaningful when `StopPolicyEnabled` is also set.

## Build the device model (on the comma 4 / with usbgpu)
```sh
# 1. export combined model (needs the torch venv)
tools/stoppolicy/.venv/bin/python -m tools.stoppolicy.export_device \
    --checkpoint runs/v1/best.pt --out /tmp/stop_device.onnx
# 2. compile to tinygrad for the device GPU
.venv/bin/python -m tools.stoppolicy.compile_device \
    --onnx /tmp/stop_device.onnx --out selfdrive/stopmodeld/models/stop_device.pkl
```

## Enable (SHADOW FIRST)
```sh
# shadow: daemon runs, planner LOGS what it would do (cloudlog "stopPolicy SHADOW"),
# zero control authority
echo -n 1 > /data/params/d/StopPolicyEnabled
# only after reviewing shadow logs on real roads, grant control (supervised):
echo -n 1 > /data/params/d/StopPolicyActive
```

## Safety invariants (enforced in code)
- Planner can only LOWER the cruise target — never raise it (decelerate-only; `apply_stop_policy` returns `min`/`0.0`).
- Deceleration uses openpilot's existing stop/MPC path → comfort-bounded by the
  planner's accel clips; no accel-limit bypass.
- Driver brake/gas overrides as always; disengagement unaffected.
- Stale/invalid `stopPolicy`, or either param off → exact stock behavior.

## Caveats
- The v1 model is **simulator-trained**; real-road performance is UNPROVEN. Run
  shadow mode and supervise every engaged drive, hands ready.
- Validated off-car: tinygrad compile + numeric parity on a CUDA dev box. The
  actual usbgpu run and on-road behavior must be validated on the comma 4.
