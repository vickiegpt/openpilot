import numpy as np
import torch

from tools.stoppolicy.export_device import build_combined, export_combined
from tools.stoppolicy.model import HIDDEN_DIM

IMG_H, IMG_W = 252, 448


def test_combined_forward_shapes():
  m = build_combined(policy_state=None, device="cpu").eval()
  img = torch.rand(1, 3, IMG_H, IMG_W)
  speed = torch.tensor([[5.0]])
  h = torch.zeros(1, HIDDEN_DIM)
  stop_logit, dist, dspeed, h_out = m(img, speed, h)
  assert stop_logit.shape == (1, 1)
  assert dist.shape == (1, 1)
  assert dspeed.shape == (1, 1)
  assert h_out.shape == (1, HIDDEN_DIM)


def test_export_and_parity(tmp_path):
  m = build_combined(policy_state=None, device="cpu").eval()
  path = tmp_path / "stop_device.onnx"
  export_combined(m, path)

  import onnxruntime as ort
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  img = np.random.rand(1, 3, IMG_H, IMG_W).astype(np.float32)
  speed = np.array([[5.0]], dtype=np.float32)
  h = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
  outs = sess.run(None, {"image": img, "speed": speed, "hidden_in": h})
  with torch.no_grad():
    t = m(torch.from_numpy(img), torch.from_numpy(speed), torch.from_numpy(h))
  for o, tt in zip(outs, t):
    np.testing.assert_allclose(o, tt.numpy(), atol=1e-3)
