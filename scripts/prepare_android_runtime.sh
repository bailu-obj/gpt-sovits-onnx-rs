#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: prepare_android_runtime.sh --output-dir PATH [--archive-dir PATH] [--qnn]"
}

OUTPUT=""
ARCHIVES=""
QNN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT="${2:?--output-dir requires a path}"; shift 2 ;;
        --archive-dir) ARCHIVES="${2:?--archive-dir requires a path}"; shift 2 ;;
        --qnn) QNN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n "$OUTPUT" ]] || { usage >&2; exit 2; }
ARCHIVES=${ARCHIVES:-"$OUTPUT/archives"}
mkdir -p "$ARCHIVES" "$OUTPUT"
MAVEN=https://repo.maven.apache.org/maven2

checksum() {
    if command -v sha256sum >/dev/null; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

fetch() {
    local group=$1 name=$2 version=$3 expected=$4 archive partial
    archive="$ARCHIVES/$name-$version.aar"
    if [[ -f "$archive" ]]; then
        [[ "$(checksum "$archive")" == "$expected" ]] ||
            { echo "Checksum mismatch: $archive" >&2; return 1; }
    else
        partial=$(mktemp "$ARCHIVES/$name-$version.XXXXXX")
        if ! curl --fail --location --retry 2 "$MAVEN/$group/$name/$version/$name-$version.aar" -o "$partial" ||
            [[ "$(checksum "$partial")" != "$expected" ]]; then
            echo "Download or checksum verification failed: $name $version ($partial)" >&2
            return 1
        fi
        mv "$partial" "$archive"
    fi
    printf '%s\n' "$archive"
}

extract() {
    local archive=$1 library=$2 partial
    partial=$(mktemp "$OUTPUT/$library.XXXXXX")
    unzip -p "$archive" "jni/arm64-v8a/$library" > "$partial"
    [[ -s "$partial" ]]
    chmod 644 "$partial"
    mv "$partial" "$OUTPUT/$library"
}

ort=$(fetch com/microsoft/onnxruntime onnxruntime-android 1.30.0 e7fb945e402205f6db858d65bb78d2bdb0812317b383976c9e3bceb4862c73f1)
extract "$ort" libonnxruntime.so
if $QNN; then
    provider=$(fetch com/qualcomm/qti onnxruntime-android-qnn 2.7.0 ecd36df222e2a947ceb13bdd6fec181d1f1841b40379e21ce519830f2fbd562a)
    runtime=$(fetch com/qualcomm/qti qnn-runtime 2.51.0 6388face1d5aa5e8958d745e5a547e92b5bec6fc332dfb1b438b1e17ffa02861)
    extract "$provider" libonnxruntime_providers_qnn.so
    for library in libQnnHtp.so libQnnHtpPrepare.so libQnnSystem.so \
        libQnnHtpV79Stub.so libQnnHtpV79Skel.so libQnnHtpV81Stub.so libQnnHtpV81Skel.so; do
        extract "$runtime" "$library"
    done
fi
echo "Prepared Android arm64 runtime in $OUTPUT"
