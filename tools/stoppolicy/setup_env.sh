#!/usr/bin/env bash
# Training env for the stop policy. Reuses miniconda's torch via system-site-packages
# (disk is nearly full; torch must not be reinstalled).
set -e
VENV=/root/openpilot/tools/stoppolicy/.venv
/opt/miniconda3/bin/python3 -m venv --system-site-packages "$VENV"
# metadrive-simulator >=0.4.2 on PyPI declares Requires-Python <3.12, but miniconda is
# Python 3.13. GitHub main (version 0.4.3) removed that cap, so install it from a pinned
# main commit instead of PyPI. Same deps otherwise; no torch/numpy conflicts observed.
METADRIVE_COMMIT=85e5dadc6c7436d324348f6e3d8f8e680c06b4db
"$VENV/bin/pip" install --no-cache-dir \
  "metadrive-simulator @ https://github.com/metadriverse/metadrive/archive/${METADRIVE_COMMIT}.tar.gz" \
  "datasets>=2.19" pillow onnx onnxruntime pyarrow pytest
if [ ! -x "$VENV/bin/pytest" ]; then
  cat > "$VENV/bin/pytest" <<'EOF'
#!/usr/bin/env bash
exec "$(dirname "$0")/python" -m pytest "$@"
EOF
  chmod +x "$VENV/bin/pytest"
fi
"$VENV/bin/python" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible from venv"
print("torch", torch.__version__, "cuda ok")
EOF
