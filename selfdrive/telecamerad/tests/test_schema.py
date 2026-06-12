from cereal import messaging
from cereal.services import SERVICE_LIST


def test_telephoto_camera_state_roundtrip():
  dat = messaging.new_message('telephotoCameraState', valid=True)
  dat.telephotoCameraState.frameId = 42
  dat.telephotoCameraState.cameraIndex = 1
  dat.telephotoCameraState.timestampEof = 123456789
  evt = messaging.log_from_bytes(dat.to_bytes())
  assert evt.telephotoCameraState.frameId == 42
  assert evt.telephotoCameraState.cameraIndex == 1
  assert evt.telephotoCameraState.timestampEof == 123456789


def test_telephoto_camera_state_service():
  svc = SERVICE_LIST['telephotoCameraState']
  assert svc.should_log
  assert svc.frequency == 40.  # 2 cameras x 20Hz on one service
