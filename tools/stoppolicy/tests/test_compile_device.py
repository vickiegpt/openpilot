import os
import subprocess

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SP_PY = os.path.join(REPO, "tools/stoppolicy/.venv/bin/python")


@pytest.fixture(scope="module")
def combined_onnx(tmp_path_factory):
  out = tmp_path_factory.mktemp("dev") / "stop_device.onnx"
  code = (
    "from tools.stoppolicy.export_device import build_combined, export_combined;"
    f"export_combined(build_combined(policy_state=None, device='cpu'), r'{out}')"
  )
  subprocess.run([SP_PY, "-c", code], cwd=REPO, check=True)
  return str(out)


@pytest.mark.slow
def test_tinygrad_parity(combined_onnx, tmp_path):
  import onnxruntime as ort
  from tools.stoppolicy.compile_device import compile_device, _rand_inputs

  tg_vals = compile_device(combined_onnx, str(tmp_path / "m.pkl"))
  sess = ort.InferenceSession(combined_onnx, providers=["CPUExecutionProvider"])
  ort_vals = sess.run(None, _rand_inputs(0))
  np.testing.assert_allclose(tg_vals[0], ort_vals[0], atol=2e-2)
  np.testing.assert_allclose(tg_vals[3], ort_vals[3], atol=2e-2)
