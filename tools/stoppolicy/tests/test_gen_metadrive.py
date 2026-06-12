import pytest

from tools.stoppolicy.data.gen_metadrive import label_for_state, plan_speed


def test_labels_green_phase():
  rec = label_for_state(ego_x=10.0, speed=12.0, light_x=100.0, light_state="green", future_speed=12.0)
  assert rec["stop_prob"] == 0.0
  assert rec["dist_to_stop"] == -1.0
  assert rec["light_state"] == "green"


def test_labels_red_before_line():
  rec = label_for_state(ego_x=50.0, speed=8.0, light_x=100.0, light_state="red", future_speed=4.0)
  assert rec["stop_prob"] == 1.0
  assert rec["dist_to_stop"] == pytest.approx(48.0)  # stop 2m before the line


def test_labels_red_after_line_is_not_a_stop():
  rec = label_for_state(ego_x=105.0, speed=12.0, light_x=100.0, light_state="red", future_speed=12.0)
  assert rec["stop_prob"] == 0.0


def test_plan_speed_decelerates_to_stop():
  assert plan_speed(dist_to_stop=80.0, cruise=12.0) == pytest.approx(12.0)
  assert 0.0 < plan_speed(dist_to_stop=10.0, cruise=12.0) < 12.0
  assert plan_speed(dist_to_stop=0.0, cruise=12.0) == 0.0


@pytest.mark.slow
def test_generate_one_episode(tmp_path):
  from tools.stoppolicy.data.gen_metadrive import generate
  eps = generate(out_dir=tmp_path, n_episodes=1, seed=0, max_total_bytes=50_000_000)
  assert len(eps) == 1
  from tools.stoppolicy.data.episode import load_episode
  ep = load_episode(eps[0])
  assert ep.meta["n_frames"] > 50
  states = {r["light_state"] for r in ep.labels}
  assert states & {"red", "green", "none"}
