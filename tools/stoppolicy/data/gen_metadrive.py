"""
MetaDrive scripted traffic-light episode generator.

Produces training episodes where the script (not perception) knows the stop
line and light state.  Three scenario kinds per seed:
  green-through  p=0.4
  red-stop-go    p=0.4
  no-light       p=0.2

Headless patches: FilterManager.render_scene_into is monkeypatched before
MetaDrive initialises to force multisamples=0, which is required on EGL/GLX
headless contexts that do not expose MSAA framebuffer support.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from tools.stoppolicy.data.episode import EpisodeWriter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRUISE = 12.0         # m/s  (~43 km/h)
STOP_MARGIN = 2.0     # stop this many metres before the stop line
COMFORT_DECEL = 2.0   # m/s² used for the braking profile
FPS = 10
IMG_W, IMG_H = 448, 252   # divisible by 14 for DINOv2

# Episode length bounds (seconds)
EP_MIN_S = 15
EP_MAX_S = 30

# Physics: 0.02 s per physics tick, 5 ticks per env.step → 0.1 s / step = 10 Hz
PHYSICS_STEP = 0.02   # s
DECISION_REPEAT = 5   # physics steps per env.step  → dt = 0.1 s = 1/FPS

# Traffic light placed this far ahead of ego spawn (metres along cumulative route)
LIGHT_DIST_MIN = 80.0
LIGHT_DIST_MAX = 120.0

# Scenario probabilities
P_GREEN = 0.4
P_RED = 0.4
# P_NONE = 0.2 (implicit)

# Speed threshold for "fully stopped"
STOP_SPEED_THRESH = 0.3   # m/s
STOP_HOLD_S = 3.0          # seconds to hold red after stopping


# ---------------------------------------------------------------------------
# Headless patch — must happen before any MetaDrive import
# ---------------------------------------------------------------------------

def _apply_headless_patches():
  """Patch Panda3D/simplepbr to work without a physical display (EGL/GLX headless)."""
  from panda3d.core import loadPrcFileData
  # Disable global MSAA (overrides the class-level setting in engine_core)
  loadPrcFileData("", "framebuffer-multisample 0")

  # FilterManager.render_scene_into silently returns None when MSAA > 0
  # is requested on a headless EGL context that doesn't support it.
  # Force 0 samples so the tonemapping quad is created successfully.
  from direct.filter.FilterManager import FilterManager as _FM

  # Skip if already patched
  if getattr(_FM, "_msaa_patched", False):
    return

  _orig_rsi = _FM.render_scene_into

  def _patched_rsi(self, *args, **kwargs):
    fbprops = kwargs.get("fbprops", None)
    if fbprops is not None:
      fbprops.set_multisamples(0)
    return _orig_rsi(self, *args, **kwargs)

  _FM.render_scene_into = _patched_rsi
  _FM._msaa_patched = True


# ---------------------------------------------------------------------------
# Pure label functions (tested by unit tests)
# ---------------------------------------------------------------------------

def label_for_state(ego_x, speed, light_x, light_state, future_speed):
  """Return a label dict for one timestep.

  ego_x      – cumulative distance along the route (m)
  speed      – current ego speed (m/s)
  light_x    – cumulative distance of the stop line (m), or None for no-light
  light_state – "green" | "red" | "yellow" | "none"
  future_speed – ego speed 1 s in the future (m/s), used as desired_speed label
  """
  must_stop = light_x is not None and light_state in ("red", "yellow") and ego_x < light_x
  dist = (light_x - STOP_MARGIN) - ego_x if must_stop else -1.0
  return {
    "speed": speed,
    "stop_prob": 1.0 if must_stop else 0.0,
    "dist_to_stop": dist,
    "desired_speed": future_speed,
    "light_state": light_state if light_x is not None else "none",
  }


def plan_speed(dist_to_stop, cruise=CRUISE):
  """Return target speed (m/s) given distance to stop line.

  Uses v = sqrt(2 * a * d) braking profile, capped at cruise.
  """
  if dist_to_stop <= 0.0:
    return 0.0
  v = (2.0 * COMFORT_DECEL * dist_to_stop) ** 0.5
  return min(cruise, v)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_env(seed: int):
  """Create a headless MetaDrive env with RGB camera at FPS=10."""
  _apply_headless_patches()

  from metadrive import MetaDriveEnv
  from metadrive.component.sensors.rgb_camera import RGBCamera

  env = MetaDriveEnv(dict(
    # reproducibility
    start_seed=seed,
    num_scenarios=1,

    # straight-ish map: three straight blocks → long straight road (~270 m total)
    map="SSS",

    # no traffic
    traffic_density=0,

    # headless offscreen
    use_render=False,
    image_observation=True,
    image_on_cuda=False,
    norm_pixel=False,  # get uint8 directly

    # camera
    sensors={"rgb_camera": (RGBCamera, IMG_W, IMG_H)},
    vehicle_config=dict(image_source="rgb_camera"),

    # physics timing: 0.02 s × 5 = 0.1 s per env.step  →  10 Hz
    physics_world_step_size=PHYSICS_STEP,
    decision_repeat=DECISION_REPEAT,

    # episode length safety net (well above EP_MAX_S)
    horizon=int(EP_MAX_S / (PHYSICS_STEP * DECISION_REPEAT)) + 100,

    # no termination on out-of-road (straight map, shouldn't happen)
    out_of_road_done=False,
    crash_vehicle_done=False,
    crash_object_done=False,

    # disable HUD noise
    show_interface=False,
    show_logo=False,
    show_fps=False,
  ))
  return env


def _get_rgb(env) -> np.ndarray:
  """Return upright RGB uint8 (H, W, 3) from the offscreen buffer.

  perceive(to_float=False) internally calls get_rgb_array_cpu() which:
    - reads the frame buffer (BGR channel order from panda3d)
    - flips the image vertically (img[::-1]) for correct orientation

  So the result is: (H, W, 3) uint8, vertically correct, BGR channel order.
  We reverse channels to produce upright RGB.
  """
  sensor = env.engine.get_sensor("rgb_camera")
  img = sensor.perceive(to_float=False)
  if not isinstance(img, np.ndarray):
    img = np.asarray(img)
  # img channels are BGR → convert to RGB
  rgb = img[..., ::-1].copy()
  return rgb.astype(np.uint8)


def _build_route_segments(env, agent):
  """Return list of (lane, odo_start, odo_end) for the agent's planned route.

  odo_start / odo_end are cumulative metres from the agent's initial spawn
  position (the start of the first lane in the route).
  """
  nav = agent.navigation
  m = env.engine.current_map
  segments = []
  odo = 0.0
  for ckpt1, ckpt2 in zip(nav.checkpoints[:-1], nav.checkpoints[1:]):
    lanes = m.road_network.graph.get(ckpt1, {}).get(ckpt2, [])
    if not lanes:
      continue
    # Use the same lane index as the agent's current lane (column index)
    lane_col = agent.lane_index[2] if agent.lane_index[2] < len(lanes) else 0
    lane = lanes[lane_col]
    odo_end = odo + lane.length
    segments.append((lane, odo, odo_end))
    odo = odo_end
  return segments


def _find_lane_at_odo(segments, target_odo):
  """Return (lane, lane_lon) for the route segment containing target_odo.

  lane_lon is the longitudinal position within that lane.
  Returns None if target_odo exceeds route length.
  """
  for lane, odo_start, odo_end in segments:
    if odo_start <= target_odo <= odo_end:
      return lane, target_odo - odo_start
  return None, None


def _spawn_light_at_odo(env, rng, target_odo, segments):
  """Spawn a BaseTrafficLight at the route position corresponding to target_odo.

  Returns (light_object, actual_odo_x) or (None, None) if placement fails.
  actual_odo_x is the odometer value where the light's stop line is.
  """
  from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight

  lane, lane_lon = _find_lane_at_odo(segments, target_odo)
  if lane is None:
    return None, None

  # Clamp lane_lon to safe bounds
  lane_lon = max(2.0, min(lane_lon, lane.length - 2.0))

  # World position at centre of lane
  pos2d = lane.position(lane_lon, 0)
  position = [float(pos2d[0]), float(pos2d[1])]

  light = env.engine.spawn_object(
    BaseTrafficLight,
    lane=lane,
    position=position,
    random_seed=int(rng.integers(0, 2**31)),
    escape_random_seed_assertion=True,
  )
  # Return the clamped odo position (odo_start + clamped lane_lon)
  # Find the odo_start for this lane to compute actual_odo_x
  for l, odo_start, odo_end in segments:
    if l is lane:
      actual_odo_x = odo_start + lane_lon
      return light, actual_odo_x
  return light, target_odo


def _throttle_from_speed_error(target_v: float, current_v: float) -> list:
  """Simple P-controller mapping speed error to [steer, throttle/brake]."""
  err = target_v - current_v
  # positive err → need to accelerate; negative → brake
  gain = 0.5
  u = float(np.clip(err * gain, -1.0, 1.0))
  return [0.0, u]  # [steering, throttle/brake]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(
  out_dir,
  n_episodes: int = 10,
  seed: int = 0,
  max_total_bytes: int = 1_000_000_000,
) -> list:
  """Generate n_episodes scripted traffic-light episodes.

  Returns list of episode directory Paths.
  """
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  rng = np.random.default_rng(seed)
  episode_dirs = []
  bytes_used = 0

  logging.basicConfig(level=logging.WARNING)

  env = _make_env(seed=seed)

  dt = PHYSICS_STEP * DECISION_REPEAT   # seconds per env.step = 1/FPS

  try:
    for ep_idx in range(n_episodes):
      if bytes_used >= max_total_bytes:
        break

      # --- Decide scenario kind ---
      r = rng.random()
      if r < P_GREEN:
        scenario = "green"
      elif r < P_GREEN + P_RED:
        scenario = "red"
      else:
        scenario = "none"

      ep_dir = out_dir / f"ep_{ep_idx:04d}"
      ep_max = max_total_bytes - bytes_used

      # Episode length in steps
      ep_steps = int(rng.integers(EP_MIN_S * FPS, EP_MAX_S * FPS + 1))

      # Reset env for each episode
      obs, info = env.reset()

      agent = env.engine.agents["default_agent"]

      # Build the route segment map for lane-accurate light placement
      segments = _build_route_segments(env, agent)

      # Odometer position where the light stop line sits
      light_odo_x = float(rng.uniform(LIGHT_DIST_MIN, LIGHT_DIST_MAX)) if scenario != "none" else None

      # Spawn light on the correct segment
      light_obj = None
      if scenario in ("green", "red") and light_odo_x is not None:
        light_obj, light_odo_x = _spawn_light_at_odo(env, rng, light_odo_x, segments)
        if light_obj is not None:
          if scenario == "green":
            light_obj.set_green()
          else:
            light_obj.set_red()
        else:
          # Couldn't place light; treat as no-light
          scenario = "none"
          light_odo_x = None

      # State machine for red scenario
      stopped_timer = 0.0   # seconds spent at standstill
      went_green = False

      # Odometer: cumulative distance driven from episode start
      odo = 0.0

      # Buffer for frame data
      frame_rgbs = []
      frame_speeds = []
      frame_odos = []
      frame_light_states = []

      done = False

      for step_i in range(ep_steps):
        if done:
          break

        current_speed = float(agent.speed)

        # Update odometer by integrating speed
        odo += current_speed * dt

        # --- Determine current scripted light state and target speed ---
        if scenario == "none":
          scripted_state = "none"
          target_v = CRUISE
        elif scenario == "green":
          scripted_state = "green"
          target_v = CRUISE
        else:  # red
          if went_green:
            scripted_state = "green"
            target_v = CRUISE
          else:
            scripted_state = "red"
            dist = (light_odo_x - STOP_MARGIN) - odo
            target_v = plan_speed(dist_to_stop=max(dist, 0.0))

            # Check if stopped
            if current_speed < STOP_SPEED_THRESH:
              stopped_timer += dt
            else:
              stopped_timer = 0.0

            if stopped_timer >= STOP_HOLD_S:
              went_green = True
              if light_obj is not None:
                light_obj.set_green()

        # --- Collect frame ---
        rgb = _get_rgb(env)
        frame_rgbs.append(rgb)
        frame_speeds.append(current_speed)
        frame_odos.append(odo)
        frame_light_states.append(scripted_state)

        # --- Control action ---
        action = _throttle_from_speed_error(target_v, current_speed)

        # --- Step simulator ---
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated

      # --- Write episode ---
      n_frames = len(frame_rgbs)
      if n_frames == 0:
        if light_obj is not None:
          env.engine.clear_objects([light_obj.id])
          light_obj = None
        continue

      with EpisodeWriter(ep_dir, source="metadrive", fps=FPS, max_bytes=ep_max) as writer:
        wrote_any = False
        for i in range(n_frames):
          speed_i = frame_speeds[i]
          odo_i = frame_odos[i]
          ls_i = frame_light_states[i]

          # future speed: speed 1 s (FPS steps) ahead, clamp at end
          future_idx = min(i + FPS, n_frames - 1)
          future_v = frame_speeds[future_idx]

          rec = label_for_state(
            ego_x=odo_i,
            speed=speed_i,
            light_x=light_odo_x,
            light_state=ls_i,
            future_speed=future_v,
          )

          ok = writer.add_frame(
            rgb=frame_rgbs[i],
            speed=rec["speed"],
            stop_prob=rec["stop_prob"],
            dist_to_stop=rec["dist_to_stop"],
            desired_speed=rec["desired_speed"],
            light_state=rec["light_state"],
          )
          if ok:
            wrote_any = True
          else:
            break  # byte budget exhausted

      if wrote_any:
        bytes_used += sum(
          p.stat().st_size
          for p in (ep_dir / "frames").glob("*.jpg")
        )
        bytes_used += (ep_dir / "labels.jsonl").stat().st_size
        bytes_used += (ep_dir / "meta.json").stat().st_size
        episode_dirs.append(ep_dir)

      # Clean up light for next episode
      if light_obj is not None:
        env.engine.clear_objects([light_obj.id])
        light_obj = None

  finally:
    env.close()

  return episode_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
  parser = argparse.ArgumentParser(description="Generate MetaDrive traffic-light episodes")
  parser.add_argument("--out", type=Path, default=Path("data/episodes"), help="Output directory")
  parser.add_argument("--n", type=int, default=10, help="Number of episodes")
  parser.add_argument("--seed", type=int, default=0, help="RNG seed")
  parser.add_argument("--max-gb", type=float, default=1.0, help="Max total output size in GB")
  args = parser.parse_args()

  eps = generate(
    out_dir=args.out,
    n_episodes=args.n,
    seed=args.seed,
    max_total_bytes=int(args.max_gb * 1e9),
  )
  print(f"Generated {len(eps)} episodes → {args.out}")
  for ep in eps:
    print(f"  {ep}")


if __name__ == "__main__":
  _cli()
