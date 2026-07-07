#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

UPSTREAM_REF="main"
FORCE=false
REMAINING=()

usage() {
    cat <<'EOF'
Usage: 01_clone.sh [options]

Clone or update upstream GPT-SoVITS into gpt-sovits-upstream (macOS only).

Options:
  --gpt-sovits-dir PATH   Clone destination (default: <repo>/gpt-sovits-upstream)
  --upstream-ref REF      Git branch/tag/commit (default: main)
  --force                 Remove existing clone and re-clone
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --gpt-sovits-dir)
        GPT_SOVITS_ROOT="$2"
        shift 2
        ;;
    --upstream-ref)
        UPSTREAM_REF="$2"
        shift 2
        ;;
    --force)
        FORCE=true
        shift
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        REMAINING+=("$1")
        shift
        ;;
    esac
done

require_macos
require_cmd git
init_paths

if $FORCE && [[ -d "${GPT_SOVITS_ROOT}" ]]; then
    log_warn "Removing existing clone: ${GPT_SOVITS_ROOT}"
    rm -rf "${GPT_SOVITS_ROOT}"
fi

if [[ -d "${GPT_SOVITS_ROOT}/.git" ]]; then
    log_info "Updating existing clone at ${GPT_SOVITS_ROOT}"
    git -C "${GPT_SOVITS_ROOT}" fetch origin "${UPSTREAM_REF}" --depth 1
    git -C "${GPT_SOVITS_ROOT}" checkout FETCH_HEAD
    log_ok "Clone updated (ref: ${UPSTREAM_REF})"
else
    log_info "Cloning GPT-SoVITS into ${GPT_SOVITS_ROOT}"
    mkdir -p "$(dirname "${GPT_SOVITS_ROOT}")"
    git clone --depth 1 --branch "${UPSTREAM_REF}" \
        https://github.com/RVC-Boss/GPT-SoVITS.git "${GPT_SOVITS_ROOT}" || \
        git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git "${GPT_SOVITS_ROOT}"
    log_ok "Clone complete"
fi
