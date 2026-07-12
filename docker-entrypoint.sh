#!/usr/bin/env bash
set -euo pipefail

OUT="${OUTPUT_PATH:-/output/results.json}"
mkdir -p "$(dirname "$OUT")"

echo "[entrypoint] video-amd starting"
echo "[entrypoint] OUTPUT_PATH=$OUT"
echo "[entrypoint] torch.cuda.is_available()=$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
echo "[entrypoint] torch.cuda.device_count()=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 'unknown')"

if [ -e /dev/kfd ]; then KFD="yes"; else KFD="no"; fi
echo "[entrypoint] /dev/kfd present? $KFD"

# Pass no input path so main._resolve_input_path() can fall back to
# /input/tasks.json, /input/input.json, or any *.json mounted in /input.
exec python -c "from main import competition_main; competition_main(None, '$OUT')"
