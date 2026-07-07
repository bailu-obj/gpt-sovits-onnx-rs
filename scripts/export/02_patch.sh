#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    cat <<'EOF'
Usage: 02_patch.sh [options]

Apply onnx-rs patch overlay onto the upstream GPT-SoVITS clone.

Options:
  --gpt-sovits-dir PATH   Upstream clone path
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --gpt-sovits-dir)
        GPT_SOVITS_ROOT="$2"
        shift 2
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        die "Unknown argument: $1"
        ;;
    esac
done

require_macos
require_cmd rsync
init_paths
require_clone

log_info "Patching ${GPT_SOVITS_ROOT}/GPT_SoVITS from ${PATCH_DIR}"
rsync -a "${PATCH_DIR}/" "${GPT_SOVITS_ROOT}/GPT_SoVITS/"
log_ok "Patch applied"
