#!/usr/bin/env bash
# Downloads every local model the agent uses (about 8 GB). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

echo "== Ollama: dialog model"
command -v ollama >/dev/null || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }
ollama pull qwen3-vl:8b-instruct

echo "== Kokoro text-to-speech"
base=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
[ -f models/kokoro-v1.0.onnx ] || curl -fL -o models/kokoro-v1.0.onnx "$base/kokoro-v1.0.onnx"
[ -f models/voices-v1.0.bin ] || curl -fL -o models/voices-v1.0.bin "$base/voices-v1.0.bin"

echo "== Ultralytics weights (official releases) and Whisper"
uv run python - <<'PY'
from pathlib import Path
from ultralytics.utils.downloads import attempt_download_asset
for name in ("yolo11x.pt", "yolov8x-worldv2.pt", "FastSAM-x.pt"):
    attempt_download_asset(str(Path("models") / name))
from faster_whisper import download_model
download_model("large-v3-turbo")  # speech, on the Apple GPU (~1.6 GB)
download_model("small")  # CPU fallback
print("ok")
PY
