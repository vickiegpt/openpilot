import numpy as np
import torch

from tools.stoppolicy.export import export_onnx
from tools.stoppolicy.model import StopPolicyHead, FEAT_DIM, HIDDEN_DIM


def test_export_and_parity(tmp_path):
  m = StopPolicyHead().eval()
  path = tmp_path / "stop_policy.onnx"
  export_onnx(m, path)

  import onnxruntime as ort
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  feat = np.random.randn(1, FEAT_DIM).astype(np.float32)
  speed = np.random.randn(1, 1).astype(np.float32)
  h = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
  ort_out = sess.run(None, {"feat": feat, "speed": speed, "hidden_in": h})
  with torch.no_grad():
    t_out, t_h = m.step(torch.from_numpy(feat), torch.from_numpy(speed), torch.from_numpy(h))
  np.testing.assert_allclose(ort_out[0], t_out["stop_logit"].numpy(), atol=1e-4)
  np.testing.assert_allclose(ort_out[3], t_h.numpy(), atol=1e-4)
