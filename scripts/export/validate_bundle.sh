#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

PYTHON=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
    --python)
        [[ $# -ge 2 ]] || die "--python requires an executable"
        PYTHON="$2"
        shift 2
        ;;
    -h|--help)
        echo "Usage: validate_bundle.sh [--python PATH] --bundle-dir PATH [--expect-v2pro] [--expect-ordinary|--expect-streaming] [--expect-qnn]"
        echo "Use an explicit Python with ONNX installed, or the export environment."
        exit 0
        ;;
    *)
        ARGS+=("$1")
        shift
        ;;
    esac
done

if [[ -n "$PYTHON" ]]; then
    "$PYTHON" "${ONNX_RS_ROOT}/scripts/validate_bundle.py" "${ARGS[@]}"
else
    run_python "${ONNX_RS_ROOT}/scripts/validate_bundle.py" "${ARGS[@]}"
fi
