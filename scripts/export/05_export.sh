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
VITS_CONV_TRANSPOSE=false
SHARED_KV=false
PREOPTIMIZE_CPU=false
MODEL_PATH=""
PROFILE="quality"
VITS_CALIBRATION_DIR=""
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
  --vits-conv-transpose   Keep transposed convolutions in the VITS upsampler
  --shared-kv             Export the T2S step decoder with reusable CPU KV buffers
  --preoptimize-cpu       Pre-fuse selected CPU graphs in the exported bundle
  --model-path PATH       Checkpoint directory (default: staged official models)
  --profile PROFILE       quality (default), compact, rtf, or mobile
  --vits-calibration-dir PATH  Mobile VITS INT8 calibration inputs (.npz)
  --ref-audio PATH        Reference WAV path for the smoke test
  --ref-text TEXT         Accurate reference transcription (required for smoke test)
  --text TEXT             Text to synthesize (required for smoke test)
  --output-dir PATH       Copy/symlink final bundle here (optional)
  --no-quant              Export floating-point models (default)
  --quant                 Apply the selected quantization profile
  --smoke-test            Run cargo gpt_sovits_demo with the supplied reference WAV and text
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --vits-only) VITS_ONLY=true; shift ;;
    --vits-conv-transpose) VITS_CONV_TRANSPOSE=true; shift ;;
    --shared-kv) SHARED_KV=true; shift ;;
    --preoptimize-cpu) PREOPTIMIZE_CPU=true; shift ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --vits-calibration-dir) VITS_CALIBRATION_DIR="$2"; shift 2 ;;
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
if $SMOKE_TEST; then
    [[ -f "${REF_AUDIO}" ]] || die "--ref-audio must name a WAV file for a smoke test"
    [[ -n "${REF_TEXT}" ]] || die "--ref-text is required for a smoke test"
    [[ -n "${TEXT}" ]] || die "--text is required for a smoke test"
    REF_AUDIO="$(cd "$(dirname "${REF_AUDIO}")" && pwd)/$(basename "${REF_AUDIO}")"
    require_cmd cargo
fi

require_macos
init_paths
require_clone
resolve_env_manager

MODEL_STAGE="${MODEL_PATH:-${GPT_SOVITS_ROOT}/models/export/${VERSION}}"
case "${PROFILE}" in
    quality|compact|rtf|mobile) ;;
    *) die "Choose a profile from quality, compact, rtf, and mobile" ;;
esac
if [[ -n "${VITS_CALIBRATION_DIR}" ]]; then
    [[ "${PROFILE}" == mobile && "${NO_QUANT}" == false ]] || die "VITS calibration requires --quant --profile mobile"
    [[ -d "${VITS_CALIBRATION_DIR}" ]] || die "Calibration directory missing: ${VITS_CALIBRATION_DIR}"
fi
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
if $VITS_CONV_TRANSPOSE; then EXPORT_ARGS+=(--vits-conv-transpose); fi
if $SHARED_KV; then EXPORT_ARGS+=(--shared-kv); fi

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
if [[ -n "${VITS_CALIBRATION_DIR}" ]]; then
    OPT_ARGS+=(--vits-calibration-dir "${VITS_CALIBRATION_DIR}")
fi

log_info "Running optimize_aio.py"
run_python "${ONNX_RS_ROOT}/scripts/optimize_aio.py" "${OPT_ARGS[@]}"

mkdir -p "${PATCHED_DIR}/g2p_en"
cp "${G2P_EN_SRC}/encoder_model.onnx" "${G2P_EN_SRC}/decoder_model.onnx" "${PATCHED_DIR}/g2p_en/"

if $PREOPTIMIZE_CPU; then
    log_info "Pre-fusing selected CPU graphs"
    run_python "${ONNX_RS_ROOT}/scripts/preoptimize_cpu.py" "${PATCHED_DIR}"
fi

VALIDATION_ARGS=(--bundle-dir "${PATCHED_DIR}")
if [[ "${VERSION}" == v2Pro || "${VERSION}" == v2ProPlus ]]; then
    VALIDATION_ARGS+=(--expect-v2pro)
fi
run_python "${ONNX_RS_ROOT}/scripts/validate_bundle.py" "${VALIDATION_ARGS[@]}"

if [[ -n "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    rsync -a --exclude='/ref.wav' --exclude='/ref.txt' \
        --exclude='/reference.wav' --exclude='/reference.txt' "${PATCHED_DIR}/" "${OUTPUT_DIR}/"
    log_ok "Copied bundle to ${OUTPUT_DIR}"
else
    log_ok "Export complete: ${PATCHED_DIR}"
fi

if $SMOKE_TEST; then
    local_model="$(cd "${OUTPUT_DIR:-${PATCHED_DIR}}" && pwd)"
    log_info "Running Rust smoke test"
    (
        cd "${ONNX_RS_ROOT}"
        cargo run --release --example gpt_sovits_demo -- \
            --model-path "${local_model}" \
            --ref-audio "${REF_AUDIO}" \
            --ref-text "${REF_TEXT}" \
            --text "${TEXT}"
    )
fi
