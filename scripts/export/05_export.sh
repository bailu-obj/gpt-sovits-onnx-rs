#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

VERSION=""
EXPORT_NAME=""
NO_QUANT=true
OUTPUT_DIR=""
SMOKE_TEST=false

usage() {
    cat <<'EOF'
Usage: 05_export.sh --version V --export-name NAME [options]

Export ONNX models and run optimize_aio post-processing.

Options:
  --version VERSION       Required: v2, v2Pro, or v2ProPlus
  --export-name NAME      Required: output bundle name (e.g. custom)
  --gpt-sovits-dir PATH   Upstream clone path
  --output-dir PATH       Copy/symlink final bundle here (optional)
  --no-quant              Disable INT8 quantization (default)
  --quant                 Enable INT8 quantization
  --smoke-test            Run cargo gpt_sovits_demo after export (needs ref.wav)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --version)
        VERSION="$2"
        shift 2
        ;;
    --export-name)
        EXPORT_NAME="$2"
        shift 2
        ;;
    --gpt-sovits-dir)
        GPT_SOVITS_ROOT="$2"
        shift 2
        ;;
    --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
    --no-quant)
        NO_QUANT=true
        shift
        ;;
    --quant)
        NO_QUANT=false
        shift
        ;;
    --smoke-test)
        SMOKE_TEST=true
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

[[ -n "${VERSION}" ]] || die "--version is required"
[[ -n "${EXPORT_NAME}" ]] || die "--export-name is required"

require_macos
init_paths
require_clone
resolve_env_manager

MODEL_STAGE="${GPT_SOVITS_ROOT}/models/export/${VERSION}"
[[ -d "${MODEL_STAGE}" ]] || die "Staged models not found at ${MODEL_STAGE}. Run 04_download_models.sh first."
[[ -f "${MODEL_STAGE}/gpt.ckpt" && -f "${MODEL_STAGE}/sovits.pth" ]] || \
    die "gpt.ckpt or sovits.pth missing in ${MODEL_STAGE}"

RAW_DIR="${GPT_SOVITS_ROOT}/onnx/${EXPORT_NAME}"
PATCHED_DIR="${GPT_SOVITS_ROOT}/onnx-patched/${EXPORT_NAME}"

log_info "Exporting ONNX (${VERSION} -> ${EXPORT_NAME})"
(
    cd "${GPT_SOVITS_ROOT}"
    run_python GPT_SoVITS/export_onnx_v2.py \
        --model_path "./models/export/${VERSION}" \
        --export_name "${EXPORT_NAME}" \
        --version "${VERSION}"
)

G2PW_ONNX="${GPT_SOVITS_ROOT}/GPT_SoVITS/text/G2PWModel/g2pW.onnx"
[[ -f "${G2PW_ONNX}" ]] || die "g2pW.onnx not found at ${G2PW_ONNX}"
cp "${G2PW_ONNX}" "${RAW_DIR}/"

G2P_EN_SRC="${GPT_SOVITS_ROOT}/models/g2p_en"
[[ -f "${G2P_EN_SRC}/encoder_model.onnx" && -f "${G2P_EN_SRC}/decoder_model.onnx" ]] || \
    die "g2p_en models not found at ${G2P_EN_SRC}. Run 04_download_models.sh first."
mkdir -p "${RAW_DIR}/g2p_en"
cp "${G2P_EN_SRC}/encoder_model.onnx" "${G2P_EN_SRC}/decoder_model.onnx" "${RAW_DIR}/g2p_en/"

OPT_ARGS=(--input-dir "${RAW_DIR}" --output-dir "${PATCHED_DIR}")
if $NO_QUANT; then
    OPT_ARGS+=(--no-quant)
fi

log_info "Running optimize_aio.py"
run_python "${ONNX_RS_ROOT}/scripts/optimize_aio.py" "${OPT_ARGS[@]}"

mkdir -p "${PATCHED_DIR}/g2p_en"
cp "${G2P_EN_SRC}/encoder_model.onnx" "${G2P_EN_SRC}/decoder_model.onnx" "${PATCHED_DIR}/g2p_en/"

if [[ -n "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    rsync -a "${PATCHED_DIR}/" "${OUTPUT_DIR}/"
    log_ok "Copied bundle to ${OUTPUT_DIR}"
else
    log_ok "Export complete: ${PATCHED_DIR}"
fi

if $SMOKE_TEST; then
    require_cmd cargo
    local_model="${OUTPUT_DIR:-${PATCHED_DIR}}"
    log_info "Running Rust smoke test"
    (
        cd "${ONNX_RS_ROOT}"
        cargo run --release --example gpt_sovits_demo -- \
            --model-path "${local_model}" \
            --ref-text "你好，这是一段参考文本。" \
            --text "你好啊，这是一个测试。吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。This demo is only for test  usage. If you find any 问题, 请修复它。"
    )
fi
