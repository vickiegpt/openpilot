"""Conftest for stopmodeld tests.

tools/stoppolicy/model.py has a top-level `import torch` which fails when torch
is not installed (e.g. on the device / in the inference-only venv).  We only
need the plain integer constants from that module, so we inject a lightweight
stub into sys.modules before any test collection touches the real file.
"""
import sys
import types


def _inject_stoppolicy_model_stub():
  """Register a minimal stub for tools.stoppolicy.model if torch is absent."""
  try:
    import torch  # noqa: F401 — if torch is present, the real module is fine
    return
  except ModuleNotFoundError:
    pass

  stub = types.ModuleType("tools.stoppolicy.model")
  stub.FEAT_DIM = 768
  stub.HIDDEN_DIM = 256

  # Ensure parent packages exist in sys.modules too
  for pkg in ("tools", "tools.stoppolicy"):
    if pkg not in sys.modules:
      sys.modules[pkg] = types.ModuleType(pkg)

  sys.modules["tools.stoppolicy.model"] = stub


_inject_stoppolicy_model_stub()
