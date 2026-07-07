#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

SKIP_ENV=false

usage() {
    cat <<'EOF'
Usage: 03_setup_env.sh [options]

Set up Python environment via conda (preferred) or uv.

Options:
  --gpt-sovits-dir PATH   Upstream clone path
  --source HF|HF-Mirror|ModelScope   Model download source for conda install.sh
  --env-manager conda|uv  Force a specific env manager
  --skip-env              Skip setup if torch+transformers already importable
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --gpt-sovits-dir)
        GPT_SOVITS_ROOT="$2"
        shift 2
        ;;
    --source)
        SOURCE="$2"
        shift 2
        ;;
    --env-manager)
        ENV_MANAGER="$2"
        shift 2
        ;;
    --skip-env)
        SKIP_ENV=true
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
init_paths
require_clone
resolve_env_manager

if $SKIP_ENV && env_is_ready; then
    log_info "Environment already ready, skipping setup"
    exit 0
fi

if [[ "${ENV_MANAGER}" == "conda" ]]; then
    require_cmd conda
    if ! conda_env_exists; then
        log_info "Creating conda env: ${CONDA_ENV_NAME}"
        conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
    fi
    log_info "Running upstream install.sh (device MPS, source ${SOURCE})"
    (
        cd "${GPT_SOVITS_ROOT}"
        conda run -n "${CONDA_ENV_NAME}" bash install.sh --device MPS --source "${SOURCE}"
    )
else
    require_cmd uv
    if ! command -v ffmpeg &>/dev/null; then
        if command -v brew &>/dev/null; then
            log_info "Installing ffmpeg via brew"
            brew install ffmpeg
        else
            die "ffmpeg not found. Install with: brew install ffmpeg"
        fi
    fi

    if [[ ! -d "${GPT_SOVITS_ROOT}/.venv" ]]; then
        log_info "Creating uv venv at ${GPT_SOVITS_ROOT}/.venv"
        uv venv --python 3.10 "${GPT_SOVITS_ROOT}/.venv"
    fi

    log_info "Installing PyTorch (CPU wheel, macOS export path)"
    UV_PROJECT_ENVIRONMENT="${GPT_SOVITS_ROOT}/.venv" uv pip install --python "${GPT_SOVITS_ROOT}/.venv/bin/python" \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu

    log_info "Installing GPT-SoVITS requirements"
    (
        cd "${GPT_SOVITS_ROOT}"
        UV_PROJECT_ENVIRONMENT="${GPT_SOVITS_ROOT}/.venv" uv pip install --python "${GPT_SOVITS_ROOT}/.venv/bin/python" \
            -r extra-req.txt --no-deps
        UV_PROJECT_ENVIRONMENT="${GPT_SOVITS_ROOT}/.venv" uv pip install --python "${GPT_SOVITS_ROOT}/.venv/bin/python" \
            -r requirements.txt
    )
fi

log_info "Installing ONNX optimize dependencies"
run_pip -r "${ONNX_RS_ROOT}/scripts/requirements.txt"

if ! env_is_ready; then
    die "Environment setup finished but import check failed (torch, transformers)"
fi

log_ok "Environment ready (${ENV_MANAGER})"
