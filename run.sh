#!/usr/bin/env bash
set -euo pipefail

VIDEO=""
CALIB=""
CONFIG_FLAG=()
GPU_FLAG=""
BACKEND_FLAG=()
DEVICE_FLAG=()
CONF_FLAG=()
IMGSZ_FLAG=()
NO_KALMAN_FLAG=""
BBOX_GT_FLAG=()
OBSERVER_VIDEO_FLAG=()
OBSERVER_JSON_FLAG=()
NO_OBSERVER_FLAG=""
DISPLAY_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video)
      VIDEO="$2"
      shift 2
      ;;
    --calib)
      CALIB="$2"
      shift 2
      ;;
    --config)
      CONFIG_FLAG=(--config "$2")
      shift 2
      ;;
    --gpu)
      GPU_FLAG="--gpu"
      shift
      ;;
    --backend)
      BACKEND_FLAG=(--backend "$2")
      shift 2
      ;;
    --device)
      DEVICE_FLAG=(--device "$2")
      shift 2
      ;;
    --conf)
      CONF_FLAG=(--conf "$2")
      shift 2
      ;;
    --imgsz)
      IMGSZ_FLAG=(--imgsz "$2")
      shift 2
      ;;
    --no-kalman)
      NO_KALMAN_FLAG="--no-kalman"
      shift
      ;;
    --bbox-gt)
      BBOX_GT_FLAG=(--bbox-gt "$2")
      shift 2
      ;;
    --observer-video)
      OBSERVER_VIDEO_FLAG=(--observer-video "$2")
      shift 2
      ;;
    --observer-json)
      OBSERVER_JSON_FLAG=(--observer-json "$2")
      shift 2
      ;;
    --no-observer)
      NO_OBSERVER_FLAG="--no-observer"
      shift
      ;;
    --display)
      DISPLAY_FLAG="--display"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$VIDEO" || -z "$CALIB" ]]; then
  echo "Usage: bash run.sh --video <path> --calib <path> [--config <path>] [--backend hybrid|auto|yolo_world] [--device cpu|cuda:0|mps|auto] [--conf <float>] [--imgsz <int>] [--gpu] [--no-kalman] [--bbox-gt <path>] [--observer-video <path>] [--observer-json <path>] [--no-observer] [--display]" >&2
  exit 1
fi

pick_python() {
  for bin in python3.11 python3.10 python3.12 python3.13 python3.14 python3; do
    if command -v "$bin" >/dev/null 2>&1; then
      if "$bin" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        command -v "$bin"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python)"
VENV_DIR=".venv"
REQ_HASH_FILE="$VENV_DIR/.requirements.sha256"
if command -v shasum >/dev/null 2>&1; then
  REQ_HASH="$(shasum -a 256 requirements.txt | awk '{print $1}')"
else
  REQ_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[run.sh] Creating virtual environment with $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

if [[ ! -f "$REQ_HASH_FILE" || "$(cat "$REQ_HASH_FILE")" != "$REQ_HASH" ]]; then
  echo "[run.sh] Installing Python dependencies"
  "$VENV_DIR/bin/python" -m pip install -q --upgrade pip
  "$VENV_DIR/bin/python" -m pip install -q -r requirements.txt
  echo "$REQ_HASH" > "$REQ_HASH_FILE"
fi

echo "[run.sh] Starting tracker"
echo "[run.sh] Video: $VIDEO"
echo "[run.sh] Calib: $CALIB"
if [[ -n "$GPU_FLAG" ]]; then
  echo "[run.sh] Mode: GPU requested"
else
  echo "[run.sh] Mode: CPU"
fi
echo ""

CMD=("$VENV_DIR/bin/python" track_bin.py --video "$VIDEO" --calib "$CALIB")
if [[ ${#CONFIG_FLAG[@]} -gt 0 ]]; then
  CMD+=("${CONFIG_FLAG[@]}")
fi
if [[ -n "$GPU_FLAG" ]]; then
  CMD+=("$GPU_FLAG")
fi
if [[ ${#BACKEND_FLAG[@]} -gt 0 ]]; then
  CMD+=("${BACKEND_FLAG[@]}")
fi
if [[ ${#DEVICE_FLAG[@]} -gt 0 ]]; then
  CMD+=("${DEVICE_FLAG[@]}")
fi
if [[ ${#CONF_FLAG[@]} -gt 0 ]]; then
  CMD+=("${CONF_FLAG[@]}")
fi
if [[ ${#IMGSZ_FLAG[@]} -gt 0 ]]; then
  CMD+=("${IMGSZ_FLAG[@]}")
fi
if [[ -n "$NO_KALMAN_FLAG" ]]; then
  CMD+=("$NO_KALMAN_FLAG")
fi
if [[ ${#BBOX_GT_FLAG[@]} -gt 0 ]]; then
  CMD+=("${BBOX_GT_FLAG[@]}")
fi
if [[ ${#OBSERVER_VIDEO_FLAG[@]} -gt 0 ]]; then
  CMD+=("${OBSERVER_VIDEO_FLAG[@]}")
fi
if [[ ${#OBSERVER_JSON_FLAG[@]} -gt 0 ]]; then
  CMD+=("${OBSERVER_JSON_FLAG[@]}")
fi
if [[ -n "$NO_OBSERVER_FLAG" ]]; then
  CMD+=("$NO_OBSERVER_FLAG")
fi
if [[ -n "$DISPLAY_FLAG" ]]; then
  CMD+=("$DISPLAY_FLAG")
fi

"${CMD[@]}"
"$VENV_DIR/bin/python" tools/review_readiness.py
