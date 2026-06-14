"""Compile the combined stop model ONNX to a tinygrad JIT pickle for on-device use.

Mirrors selfdrive/modeld/compile_modeld.py at small scale: OnnxRunner -> TinyJit ->
capture/replay -> pickle. Runs on Device.DEFAULT (NV on a dev box, the usbgpu on
the comma device).

Adaptation notes vs the spec template:
- OnnxRunner takes a path directly (str or pathlib.Path), not a pre-loaded model object.
- OnnxRunner.__call__ returns a dict keyed by ONNX output names, not a list.
  We preserve ONNX declaration order via runner.graph_outputs:
  [stop_logit, dist_to_stop, desired_speed, hidden_out].
- Device.default (lowercase) is the property that returns the active device object;
  Device.DEFAULT (uppercase) is the string name ("NV"). We use Device.default for
  synchronize() and Device.DEFAULT for Tensor construction device arg.
- Tensor inputs are passed as numpy arrays to OnnxRunner; OnnxRunner wraps them
  internally. The JIT-wrapped closure accepts Tensors and forwards them.

NVRTC note (dev box): this system has NVRTC 12.0 on the default library path, but the
GPU is an NVIDIA Blackwell (sm_120) which requires NVRTC >= 12.8. Tinygrad exposes the
NVRTC_PATH env-var override via tinygrad.runtime.support.c.DLL. We probe for the newer
library at the known CUDA 12.8 install location and set it before tinygrad loads nvrtc.
On the comma device this is a non-issue (shipped CUDA matches the GPU).
"""
import argparse
import os
import pickle
import time


def _patch_nvrtc_path_if_needed():
  """Point tinygrad at NVRTC 12.8 when the system default is too old for this GPU.

  Must run before any tinygrad import that triggers nvrtc lazy-load.
  No-op if NVRTC_PATH is already set or the newer library is not present.
  """
  if os.environ.get("NVRTC_PATH"):
    return
  candidate = "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so"
  if os.path.isfile(candidate):
    os.environ["NVRTC_PATH"] = candidate


_patch_nvrtc_path_if_needed()

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.onnx import OnnxRunner

IMG_H, IMG_W, HIDDEN_DIM = 252, 448, 256


def _rand_inputs(seed):
  rng = np.random.default_rng(seed)
  return {
    "image": rng.random((1, 3, IMG_H, IMG_W), dtype=np.float32),
    "speed": rng.random((1, 1), dtype=np.float32),
    "hidden_in": np.zeros((1, HIDDEN_DIM), dtype=np.float32),
  }


def build_runner(onnx_path):
  # OnnxRunner accepts a path (str / pathlib.Path) and parses the ONNX itself.
  runner = OnnxRunner(onnx_path)

  # OnnxRunner.__call__ returns a dict {output_name: Tensor}.
  # We sort by name to get a deterministic tuple order:
  #   dist_to_stop, desired_speed, hidden_out, stop_logit (alphabetical)
  # The test checks index 0 (stop_logit) and index 3 (hidden_out) against ort,
  # so we preserve the ONNX declaration order by reading graph_outputs directly.
  output_names = runner.graph_outputs  # tuple in ONNX declaration order

  def run(image, speed, hidden_in):
    outs = runner({"image": image, "speed": speed, "hidden_in": hidden_in})
    return tuple(outs[name].cast("float32") for name in output_names)

  return TinyJit(run, prune=True)


def compile_device(onnx_path, out_path):
  jit = build_runner(onnx_path)
  last = None
  for i in range(3):
    inp = {k: Tensor(v, device=Device.DEFAULT) for k, v in _rand_inputs(0).items()}
    Device.default.synchronize()
    st = time.perf_counter()
    outs = jit(**inp)
    vals = [o.numpy() for o in outs]
    Device.default.synchronize()
    print(f"  [{i+1}/3] {(time.perf_counter()-st)*1e3:.1f} ms")
    last = vals
  jit = pickle.loads(pickle.dumps(jit))
  with open(out_path, "wb") as f:
    pickle.dump(jit, f)
  print(f"saved {out_path}")
  return last


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--onnx", required=True)
  p.add_argument("--out", required=True)
  args = p.parse_args()
  compile_device(args.onnx, args.out)


if __name__ == "__main__":
  main()
