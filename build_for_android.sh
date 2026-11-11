#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/target/android"
ARCHIVES=""
QNN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUT="${2:?--output-dir requires a path}"; shift 2 ;;
        --archive-dir) ARCHIVES="${2:?--archive-dir requires a path}"; shift 2 ;;
        --qnn) QNN=true; shift ;;
        -h|--help) echo "Usage: build_for_android.sh [--output-dir PATH] [--archive-dir PATH] [--qnn]"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

NDK=${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}
[[ -d "$NDK" ]] || { echo "Set ANDROID_NDK_ROOT or ANDROID_NDK_HOME to the Android NDK directory" >&2; exit 1; }
case "$(uname -s)" in
    Darwin) HOST_TAG=darwin-x86_64 ;;
    Linux) HOST_TAG=linux-x86_64 ;;
    *) echo "Build on macOS or Linux" >&2; exit 1 ;;
esac
TARGET=aarch64-linux-android
API=28
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG"
LINKER="$TOOLCHAIN/bin/$TARGET$API-clang"
[[ -x "$LINKER" ]] || { echo "Android linker not found: $LINKER" >&2; exit 1; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
ARCHIVES=${ARCHIVES:-"$OUT/archives"}
LIBRARIES="$OUT/lib"
RUNTIME_ARGS=(--output-dir "$LIBRARIES" --archive-dir "$ARCHIVES")
if $QNN; then RUNTIME_ARGS+=(--qnn); fi
"$ROOT/scripts/prepare_android_runtime.sh" "${RUNTIME_ARGS[@]}"
cp "$TOOLCHAIN/sysroot/usr/lib/$TARGET/libc++_shared.so" "$LIBRARIES/"

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$OUT/cargo"}
export ANDROID_NDK_ROOT="$NDK" ANDROID_NDK_HOME="$NDK" ANDROID_NDK="$NDK"
export ANDROID_PLATFORM="android-$API"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$LINKER"
export CC_aarch64_linux_android="$LINKER"
export CXX_aarch64_linux_android="$TOOLCHAIN/bin/$TARGET$API-clang++"
export AR_aarch64_linux_android="$TOOLCHAIN/bin/llvm-ar"
cat > "$OUT/android-toolchain.cmake" <<EOF
set(ANDROID_ABI arm64-v8a CACHE STRING "" FORCE)
set(ANDROID_PLATFORM android-$API CACHE STRING "" FORCE)
include("$NDK/build/cmake/android.toolchain.cmake")
EOF
export CMAKE_TOOLCHAIN_FILE_aarch64_linux_android="$OUT/android-toolchain.cmake"
export ORT_LIB_LOCATION="$LIBRARIES" ORT_PREFER_DYNAMIC_LINK=1
export CARGO_ENCODED_RUSTFLAGS=$'-C\x1ftarget-cpu=generic\x1f-C\x1flink-arg=-Wl,-z,max-page-size=16384\x1f-C\x1flink-arg=-Wl,-z,common-page-size=16384'
cargo build --manifest-path "$ROOT/Cargo.toml" --target "$TARGET" --release \
    --features jni,shared-ort-pool --example gpt_sovits_demo --example gpt_sovits_demo_jni
BIN="$CARGO_TARGET_DIR/$TARGET/release/examples"
cp "$BIN/gpt_sovits_demo" "$OUT/"
cp "$BIN/libgpt_sovits_demo_jni.so" "$LIBRARIES/"
echo "Android CLI: $OUT/gpt_sovits_demo"
echo "Android libraries: $LIBRARIES"
