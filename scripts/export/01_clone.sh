#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

UPSTREAM_REF="main"
FORCE=false

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
        die "Unknown argument: $1"
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

if [[ ! -e "${GPT_SOVITS_ROOT}/.git" ]]; then
    mkdir -p "${GPT_SOVITS_ROOT}"
    [[ -z "$(ls -A "${GPT_SOVITS_ROOT}")" ]] || die "Clone destination must be empty"
    git init -q "${GPT_SOVITS_ROOT}"
    git -C "${GPT_SOVITS_ROOT}" remote add origin https://github.com/RVC-Boss/GPT-SoVITS.git
fi

log_info "Fetching GPT-SoVITS ref ${UPSTREAM_REF} into ${GPT_SOVITS_ROOT}"
git -C "${GPT_SOVITS_ROOT}" fetch --depth=1 -- origin "${UPSTREAM_REF}"
git -C "${GPT_SOVITS_ROOT}" checkout --detach FETCH_HEAD
log_ok "Upstream commit: $(git -C "${GPT_SOVITS_ROOT}" rev-parse HEAD)"
