from openpilot.selfdrive.controls.lib.longitudinal_planner import apply_stop_policy


def test_disabled_is_noop():
  v = apply_stop_policy(20.0, enabled=False, active=False, fresh=True,
                        model_valid=True, should_stop=True, desired_speed=0.0)
  assert v == 20.0


def test_shadow_does_not_change_v_cruise():
  v = apply_stop_policy(20.0, enabled=True, active=False, fresh=True,
                        model_valid=True, should_stop=True, desired_speed=0.0)
  assert v == 20.0


def test_active_should_stop_forces_zero():
  v = apply_stop_policy(20.0, enabled=True, active=True, fresh=True,
                        model_valid=True, should_stop=True, desired_speed=3.0)
  assert v == 0.0


def test_active_lowers_to_desired_speed():
  v = apply_stop_policy(20.0, enabled=True, active=True, fresh=True,
                        model_valid=True, should_stop=False, desired_speed=8.0)
  assert v == 8.0


def test_never_raises_v_cruise():
  v = apply_stop_policy(5.0, enabled=True, active=True, fresh=True,
                        model_valid=True, should_stop=False, desired_speed=18.0)
  assert v == 5.0


def test_stale_or_invalid_is_noop():
  assert apply_stop_policy(15.0, enabled=True, active=True, fresh=False,
                           model_valid=True, should_stop=True, desired_speed=0.0) == 15.0
  assert apply_stop_policy(15.0, enabled=True, active=True, fresh=True,
                           model_valid=False, should_stop=True, desired_speed=0.0) == 15.0
