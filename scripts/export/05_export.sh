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
VITS_ONLY=false
MODEL_PATH=""
PROFILE="quality"
REF_AUDIO=""
REF_TEXT=""
TEXT=""

usage() {
    cat <<'EOF'
Usage: 05_export.sh --version V --export-name NAME [options]

Export ONNX models and run optimize_aio post-processing.

Options:
  --version VERSION       Required: v2, v2Pro, or v2ProPlus
  --export-name NAME      Required: output bundle name (e.g. v2pro)
  --gpt-sovits-dir PATH   Upstream clone path
  --vits-only             Refresh VITS only; existing speech/text exports required
  --model-path PATH       Checkpoint directory (default: staged official models)
  --profile PROFILE       quality (default) or compact quantization
  --ref-audio PATH        Reference WAV to copy into the bundle
  --ref-text TEXT         Accurate reference transcription (required for smoke test)
  --text TEXT             Text to synthesize (required for smoke test)
  --output-dir PATH       Copy/symlink final bundle here (optional)
  --no-quant              Disable INT8 quantization (default)
  --quant                 Enable INT8 quantization
  --smoke-test            Run cargo gpt_sovits_demo after export (needs ref.wav)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --vits-only) VITS_ONLY=true; shift ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --ref-audio) REF_AUDIO="$2"; shift 2 ;;
    --ref-text) REF_TEXT="$2"; shift 2 ;;
    --text) TEXT="$2"; shift 2 ;;
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

MODEL_STAGE="${MODEL_PATH:-${GPT_SOVITS_ROOT}/models/export/${VERSION}}"
[[ "${PROFILE}" == quality || "${PROFILE}" == compact ]] || die "Invalid profile"
GPT_CHECKPOINT="${MODEL_STAGE}/gpt.ckpt"
SOVITS_CHECKPOINT="${MODEL_STAGE}/sovits.pth"
if [[ ! -f "${GPT_CHECKPOINT}" && -f "${MODEL_STAGE}/kaoyu_gpt.ckpt" ]]; then
    GPT_CHECKPOINT="${MODEL_STAGE}/kaoyu_gpt.ckpt"
    SOVITS_CHECKPOINT="${MODEL_STAGE}/kaoyu_sovits.pth"
fi
[[ -d "${MODEL_STAGE}" ]] || die "Staged models not found at ${MODEL_STAGE}. Run 04_download_models.sh first."
[[ -f "${GPT_CHECKPOINT}" && -f "${SOVITS_CHECKPOINT}" ]] || \
    die "gpt.ckpt or sovits.pth missing in ${MODEL_STAGE}"

RAW_DIR="${GPT_SOVITS_ROOT}/onnx/${EXPORT_NAME}"
PATCHED_DIR="${GPT_SOVITS_ROOT}/onnx-patched/${EXPORT_NAME}"

EXPORT_ARGS=(--version "${VERSION}")
if $VITS_ONLY; then EXPORT_ARGS+=(--vits-only); fi

log_info "Exporting ONNX (${VERSION} -> ${EXPORT_NAME})"
(
    cd "${GPT_SOVITS_ROOT}"
    run_python GPT_SoVITS/export_onnx_v2.py \
        --model_path "${MODEL_STAGE}" \
        --gpt-checkpoint "${GPT_CHECKPOINT}" --sovits-checkpoint "${SOVITS_CHECKPOINT}" \
        --export_name "${EXPORT_NAME}" \
        "${EXPORT_ARGS[@]}"
)

G2PW_ONNX="${GPT_SOVITS_ROOT}/GPT_SoVITS/text/G2PWModel/g2pW.onnx"
[[ -f "${G2PW_ONNX}" ]] || die "g2pW.onnx not found at ${G2PW_ONNX}"
cp "${G2PW_ONNX}" "${RAW_DIR}/"

G2P_EN_SRC="${GPT_SOVITS_ROOT}/models/g2p_en"
[[ -f "${G2P_EN_SRC}/encoder_model.onnx" && -f "${G2P_EN_SRC}/decoder_model.onnx" ]] || \
    die "g2p_en models not found at ${G2P_EN_SRC}. Run 04_download_models.sh first."
mkdir -p "${RAW_DIR}/g2p_en"
cp "${G2P_EN_SRC}/encoder_model.onnx" "${G2P_EN_SRC}/decoder_model.onnx" "${RAW_DIR}/g2p_en/"

OPT_ARGS=(--input-dir "${RAW_DIR}" --output-dir "${PATCHED_DIR}" --profile "${PROFILE}")
if $NO_QUANT; then
    OPT_ARGS+=(--no-quant)
fi

log_info "Running optimize_aio.py"
run_python "${ONNX_RS_ROOT}/scripts/optimize_aio.py" "${OPT_ARGS[@]}"

mkdir -p "${PATCHED_DIR}/g2p_en"
cp "${G2P_EN_SRC}/encoder_model.onnx" "${G2P_EN_SRC}/decoder_model.onnx" "${PATCHED_DIR}/g2p_en/"

if [[ -n "${REF_AUDIO}" ]]; then
    cp "${REF_AUDIO}" "${RAW_DIR}/ref.wav"
    cp "${REF_AUDIO}" "${PATCHED_DIR}/ref.wav"
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    rsync -a "${PATCHED_DIR}/" "${OUTPUT_DIR}/"
    log_ok "Copied bundle to ${OUTPUT_DIR}"
else
    log_ok "Export complete: ${PATCHED_DIR}"
fi

if $SMOKE_TEST; then
    [[ -n "${REF_TEXT}" ]] || die "--ref-text is required for a smoke test"
    [[ -n "${TEXT}" ]] || die "--text is required for a smoke test"
    require_cmd cargo
    local_model="${OUTPUT_DIR:-${PATCHED_DIR}}"
    log_info "Running Rust smoke test"
    (
        cd "${ONNX_RS_ROOT}"
        cargo run --release --example gpt_sovits_demo -- \
            --model-path "${local_model}" \
            --ref-text "${REF_TEXT}" \
            --text "${TEXT}"
    )
fi
