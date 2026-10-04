#!/usr/bin/env bash
# Validate an exported ONNX bundle (v2 / v2Pro / v2ProPlus).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/common.sh
source "${SCRIPT_DIR}/lib/common.sh"

BUNDLE_DIR=""
EXPECT_V2PRO=false

usage() {
    cat <<'EOF'
Usage: validate_bundle.sh --bundle-dir PATH [--expect-v2pro]

Checks required ONNX artifacts and Pro-family metadata in an export bundle.

Options:
  --bundle-dir PATH   Required: onnx-patched export directory
  --expect-v2pro      Require sv.onnx and sv_emb in VITS graph
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --bundle-dir)
        BUNDLE_DIR="$2"
        shift 2
        ;;
    --expect-v2pro)
        EXPECT_V2PRO=true
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

[[ -n "${BUNDLE_DIR}" ]] || die "--bundle-dir is required"
[[ -d "${BUNDLE_DIR}" ]] || die "Bundle directory not found: ${BUNDLE_DIR}"

require_file() {
    [[ -f "${BUNDLE_DIR}/$1" ]] || die "Missing required file: ${BUNDLE_DIR}/$1"
}

PREFIX=""
for candidate in "${BUNDLE_DIR}"/*_vits.onnx; do
    [[ -e "${candidate}" ]] || continue
    PREFIX="$(basename "${candidate}" "_vits.onnx")"
    break
done
[[ -n "${PREFIX}" ]] || die "No *_vits.onnx found in ${BUNDLE_DIR}"

require_file "ssl.onnx"
require_file "bert.onnx"
require_file "g2pW.onnx"
require_file "${PREFIX}_vits.onnx"
require_file "${PREFIX}_t2s_encoder.onnx"
require_file "${PREFIX}_t2s_fs_decoder.onnx"
require_file "${PREFIX}_t2s_s_decoder.onnx"
require_file "g2p_en/encoder_model.onnx"
require_file "g2p_en/decoder_model.onnx"

resolve_env_manager
run_python "${ONNX_RS_ROOT}/scripts/vits_dynamic.py" "${BUNDLE_DIR}/${PREFIX}_vits.onnx"

if $EXPECT_V2PRO; then
    require_file "sv.onnx"
    grep -aq "sv_emb" "${BUNDLE_DIR}/${PREFIX}_vits.onnx" || \
        die "Expected sv_emb input in ${PREFIX}_vits.onnx"
fi

JSON_CANDIDATE="$(dirname "${BUNDLE_DIR}")/${PREFIX}.json"
if [[ -f "${JSON_CANDIDATE}" ]]; then
    log_info "Metadata: ${JSON_CANDIDATE}"
    grep -E '"Version"|"IsV2Pro"' "${JSON_CANDIDATE}" || true
fi

log_ok "Bundle validation passed for ${BUNDLE_DIR} (prefix=${PREFIX})"
