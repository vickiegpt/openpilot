from cereal import messaging
from cereal.services import SERVICE_LIST


def test_stop_policy_roundtrip():
  dat = messaging.new_message('stopPolicy', valid=True)
  dat.stopPolicy.shouldStop = True
  dat.stopPolicy.desiredSpeed = 3.5
  dat.stopPolicy.distToStop = 12.0
  dat.stopPolicy.stopProb = 0.91
  dat.stopPolicy.modelValid = True
  dat.stopPolicy.frameId = 7
  evt = messaging.log_from_bytes(dat.to_bytes())
  assert evt.stopPolicy.shouldStop is True
  assert abs(evt.stopPolicy.desiredSpeed - 3.5) < 1e-5
  assert abs(evt.stopPolicy.distToStop - 12.0) < 1e-5
  assert abs(evt.stopPolicy.stopProb - 0.91) < 1e-5
  assert evt.stopPolicy.modelValid is True
  assert evt.stopPolicy.frameId == 7


def test_stop_policy_service():
  svc = SERVICE_LIST['stopPolicy']
  assert svc.should_log
  assert svc.frequency == 20.
