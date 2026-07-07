#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

VERSION=""

usage() {
    cat <<'EOF'
Usage: 04_download_models.sh --version v2|v2Pro|v2ProPlus [options]

Download pretrained assets, G2PW, g2p_en ONNX, and stage gpt.ckpt + sovits.pth for export.

Options:
  --version VERSION       Required: v2, v2Pro, or v2ProPlus
  --gpt-sovits-dir PATH   Upstream clone path
  --source HF|HF-Mirror|ModelScope
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --version)
        VERSION="$2"
        shift 2
        ;;
    --gpt-sovits-dir)
        GPT_SOVITS_ROOT="$2"
        shift 2
        ;;
    --source)
        SOURCE="$2"
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

[[ -n "${VERSION}" ]] || die "--version is required"

require_macos
require_cmd unzip
init_paths
require_clone
set_download_urls
resolve_env_manager

PRETRAINED_DIR="${GPT_SOVITS_ROOT}/GPT_SoVITS/pretrained_models"
G2PW_DIR="${GPT_SOVITS_ROOT}/GPT_SoVITS/text/G2PWModel"
G2P_EN_DIR="${GPT_SOVITS_ROOT}/models/g2p_en"

download_pretrained_bundle() {
    if [[ -d "${PRETRAINED_DIR}/sv" ]]; then
        log_info "Pretrained bundle already present"
        return
    fi
    log_info "Downloading pretrained_models.zip"
    local zip="${GPT_SOVITS_ROOT}/pretrained_models.zip"
    download_file "${PRETRAINED_URL}" "${zip}"
    unzip -q -o "${zip}" -d "${GPT_SOVITS_ROOT}/GPT_SoVITS"
    rm -f "${zip}"
    log_ok "Pretrained bundle extracted"
}

download_g2pw() {
    if [[ -d "${G2PW_DIR}" ]]; then
        log_info "G2PWModel already present"
        return
    fi
    log_info "Downloading G2PWModel.zip"
    local zip="${GPT_SOVITS_ROOT}/G2PWModel.zip"
    download_file "${G2PW_URL}" "${zip}"
    unzip -q -o "${zip}" -d "${GPT_SOVITS_ROOT}/GPT_SoVITS/text"
    rm -f "${zip}"
    log_ok "G2PWModel extracted"
}

download_g2p_en() {
    if [[ -f "${G2P_EN_DIR}/encoder_model.onnx" && -f "${G2P_EN_DIR}/decoder_model.onnx" ]]; then
        log_info "g2p_en ONNX models already present"
        return
    fi
    mkdir -p "${G2P_EN_DIR}"
    log_info "Downloading g2p_en ONNX models (cisco-ai/mini-bart-g2p)"
    download_hf_file \
        "cisco-ai/mini-bart-g2p/resolve/main/onnx/encoder_model.onnx" \
        "${G2P_EN_DIR}/encoder_model.onnx"
    download_hf_file \
        "cisco-ai/mini-bart-g2p/resolve/main/onnx/decoder_model.onnx" \
        "${G2P_EN_DIR}/decoder_model.onnx"
    log_ok "g2p_en ONNX models downloaded to ${G2P_EN_DIR}"
}

download_hf_file() {
    local repo_path="$1"
    local dest="$2"
    if [[ -f "${dest}" ]]; then
        return
    fi
    mkdir -p "$(dirname "${dest}")"
    log_info "Downloading $(basename "${dest}")"
    download_file "$(hf_resolve_url "${repo_path}")" "${dest}"
}

download_v2pro_extras() {
    case "${VERSION}" in
    v2Pro | v2ProPlus)
        download_hf_file "lj1995/GPT-SoVITS/resolve/main/s1v3.ckpt" "${PRETRAINED_DIR}/s1v3.ckpt"
        download_hf_file "lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2Pro.pth" "${PRETRAINED_DIR}/v2Pro/s2Gv2Pro.pth"
        download_hf_file "lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2ProPlus.pth" "${PRETRAINED_DIR}/v2Pro/s2Gv2ProPlus.pth"
        download_hf_file "lj1995/GPT-SoVITS/resolve/main/sv/pretrained_eres2netv2w24s4ep4.ckpt" \
            "${PRETRAINED_DIR}/sv/pretrained_eres2netv2w24s4ep4.ckpt"
        ;;
    esac
}

download_uv_language_assets() {
    if [[ "${ENV_MANAGER}" != "uv" ]]; then
        return
    fi

    local py_prefix
    py_prefix="$(run_python -c "import sys; print(sys.prefix)")"

    if [[ ! -d "${py_prefix}/nltk_data" ]]; then
        log_info "Downloading NLTK data for uv env"
        local zip="${GPT_SOVITS_ROOT}/nltk_data.zip"
        download_file "${NLTK_URL}" "${zip}"
        unzip -q -o "${zip}" -d "${py_prefix}"
        rm -f "${zip}"
    fi

    local pyopenjtalk_dir
    pyopenjtalk_dir="$(run_python -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))" 2>/dev/null || true)"
    if [[ -n "${pyopenjtalk_dir}" && ! -d "${pyopenjtalk_dir}/open_jtalk_dic" ]]; then
        log_info "Downloading OpenJTalk dictionary for uv env"
        local tgz="${GPT_SOVITS_ROOT}/open_jtalk_dic_utf_8-1.11.tar.gz"
        download_file "${PYOPENJTALK_URL}" "${tgz}"
        tar -xzf "${tgz}" -C "${pyopenjtalk_dir}"
        rm -f "${tgz}"
    fi
}

stage_export_models() {
    version_weight_paths "${VERSION}"
    [[ -f "${GPT_SRC}" ]] || die "GPT weights not found: ${GPT_SRC}"
    [[ -f "${SOVITS_SRC}" ]] || die "SoVITS weights not found: ${SOVITS_SRC}"

    local stage_dir="${GPT_SOVITS_ROOT}/models/export/${VERSION}"
    mkdir -p "${stage_dir}"

    ln -sfn "${GPT_SRC}" "${stage_dir}/gpt.ckpt"
    ln -sfn "${SOVITS_SRC}" "${stage_dir}/sovits.pth"
    log_ok "Staged export models at ${stage_dir}"
}

download_pretrained_bundle
download_g2pw
download_g2p_en
download_v2pro_extras
download_uv_language_assets
stage_export_models

log_ok "Model download and staging complete for ${VERSION}"
