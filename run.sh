#!/usr/bin/env bash
# Launcher for FaceFusion on this machine (native venv, no conda).
# onnxruntime-gpu 1.26 needs CUDA 12 + cuDNN 9, supplied by the pip
# nvidia-*-cu12 wheels, so they must be on LD_LIBRARY_PATH.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/venv/bin/python"
SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"

export LD_LIBRARY_PATH="$(ls -d "$SITE"/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"

cd "$HERE"
# Default to the Gradio UI; pass e.g. `headless-run ...` to override.
exec "$PY" facefusion.py "${@:-run}"
