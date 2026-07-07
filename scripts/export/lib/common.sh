#!/usr/bin/env bash
# Shared helpers for macOS GPT-SoVITS ONNX export pipeline.

set -euo pipefail

RESET="\033[0m"
BOLD="\033[1m"
ERROR="\033[1;31m[ERROR]:${RESET} "
WARNING="\033[1;33m[WARNING]:${RESET} "
INFO="\033[1;32m[INFO]:${RESET} "
SUCCESS="\033[1;34m[SUCCESS]:${RESET} "

EXPORT_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_DIR="$(cd "${EXPORT_LIB_DIR}/.." && pwd)"
ONNX_RS_ROOT="$(cd "${EXPORT_DIR}/../.." && pwd)"
PATCH_DIR="${ONNX_RS_ROOT}/patch/GPT_SoVITS"

GPT_SOVITS_ROOT="${GPT_SOVITS_ROOT:-${ONNX_RS_ROOT}/gpt-sovits-upstream}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-GPTSoVits}"
SOURCE="${SOURCE:-HF}"
ENV_MANAGER="${ENV_MANAGER:-}"

die() {
    echo -e "${ERROR}$*" >&2
    exit 1
}

log_info() { echo -e "${INFO}$*" >&2; }
log_warn() { echo -e "${WARNING}$*" >&2; }
log_ok() { echo -e "${SUCCESS}$*" >&2; }

require_macos() {
    [[ "$(uname -s)" == "Darwin" ]] || die "This script supports macOS only."
}

require_cmd() {
    local cmd="$1"
    command -v "$cmd" &>/dev/null || die "Required command not found: ${cmd}"
}

init_paths() {
    GPT_SOVITS_ROOT="$(cd "${GPT_SOVITS_ROOT}" 2>/dev/null && pwd || echo "${GPT_SOVITS_ROOT}")"
    [[ -d "${PATCH_DIR}" ]] || die "Patch directory not found: ${PATCH_DIR}"
}

detect_env_manager() {
    if [[ -n "${ENV_MANAGER}" ]]; then
        echo "${ENV_MANAGER}"
        return
    fi

    local has_conda=false has_uv=false
    command -v conda &>/dev/null && has_conda=true
    command -v uv &>/dev/null && has_uv=true

    if $has_conda; then
        echo conda
    elif $has_uv; then
        echo uv
    else
        die "Need conda or uv. Install Miniconda or: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
}

resolve_env_manager() {
    if [[ -n "${ENV_MANAGER}" ]]; then
        return
    fi
    ENV_MANAGER="$(detect_env_manager)"
    log_info "Using env manager: ${ENV_MANAGER}"
}

set_download_urls() {
    case "${SOURCE}" in
    HF)
        PRETRAINED_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
        G2PW_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
        NLTK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
        PYOPENJTALK_URL="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
        HF_BASE="https://huggingface.co"
        ;;
    HF-Mirror)
        PRETRAINED_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip"
        G2PW_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip"
        NLTK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/nltk_data.zip"
        PYOPENJTALK_URL="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/open_jtalk_dic_utf_8-1.11.tar.gz"
        HF_BASE="https://hf-mirror.com"
        ;;
    ModelScope)
        PRETRAINED_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip"
        G2PW_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
        NLTK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/nltk_data.zip"
        PYOPENJTALK_URL="https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/open_jtalk_dic_utf_8-1.11.tar.gz"
        HF_BASE="https://huggingface.co"
        ;;
    *)
        die "Invalid --source: ${SOURCE} (use HF, HF-Mirror, or ModelScope)"
        ;;
    esac
}

download_file() {
    local url="$1"
    local out="$2"
    if command -v curl &>/dev/null; then
        curl -fL --retry 3 --retry-delay 5 -o "${out}" "${url}"
    elif command -v wget &>/dev/null; then
        wget -q --show-progress -O "${out}" "${url}"
    else
        die "Need curl or wget to download files"
    fi
}

hf_resolve_url() {
    local repo_path="$1"
    echo "${HF_BASE}/${repo_path}"
}

conda_env_exists() {
    conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"
}

env_is_ready() {
    resolve_env_manager
    if [[ "${ENV_MANAGER}" == "conda" ]]; then
        conda run -n "${CONDA_ENV_NAME}" python -c "import torch, transformers" &>/dev/null
    else
        [[ -d "${GPT_SOVITS_ROOT}/.venv" ]] || return 1
        UV_PROJECT_ENVIRONMENT="${GPT_SOVITS_ROOT}/.venv" uv run --directory "${GPT_SOVITS_ROOT}" \
            python -c "import torch, transformers" &>/dev/null
    fi
}

run_python() {
    resolve_env_manager
    if [[ "${ENV_MANAGER}" == "conda" ]]; then
        conda run -n "${CONDA_ENV_NAME}" python "$@"
    else
        UV_PROJECT_ENVIRONMENT="${GPT_SOVITS_ROOT}/.venv" uv run --directory "${GPT_SOVITS_ROOT}" python "$@"
    fi
}

run_pip() {
    # Args are passed to `pip install` (do not include the word "install").
    resolve_env_manager
    if [[ "${ENV_MANAGER}" == "conda" ]]; then
        conda run -n "${CONDA_ENV_NAME}" pip install "$@"
    else
        UV_PROJECT_ENVIRONMENT="${GPT_SOVITS_ROOT}/.venv" uv pip install --python "${GPT_SOVITS_ROOT}/.venv/bin/python" "$@"
    fi
}

require_clone() {
    [[ -d "${GPT_SOVITS_ROOT}/.git" ]] || die "Upstream clone not found at ${GPT_SOVITS_ROOT}. Run 01_clone.sh first."
}

parse_common_args() {
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
        *)
            return 0
            ;;
        esac
    done
}

version_weight_paths() {
    local version="$1"
    local pretrained="${GPT_SOVITS_ROOT}/GPT_SoVITS/pretrained_models"
    case "${version}" in
    v2)
        GPT_SRC="${pretrained}/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        SOVITS_SRC="${pretrained}/gsv-v2final-pretrained/s2G2333k.pth"
        ;;
    v2Pro)
        GPT_SRC="${pretrained}/s1v3.ckpt"
        SOVITS_SRC="${pretrained}/v2Pro/s2Gv2Pro.pth"
        ;;
    v2ProPlus)
        GPT_SRC="${pretrained}/s1v3.ckpt"
        SOVITS_SRC="${pretrained}/v2Pro/s2Gv2ProPlus.pth"
        ;;
    *)
        die "Invalid --version: ${version} (use v2, v2Pro, or v2ProPlus)"
        ;;
    esac
}
