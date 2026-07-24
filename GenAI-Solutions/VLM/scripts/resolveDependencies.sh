#============================================================================
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DEVICE_BUNDLE_PATH="/data/local/tmp/genie_bundle"

# ── Helpers ──────────────────────────────────────────────────────────────────

red()    { echo -e "\033[0;31m$*\033[0m"; }
green()  { echo -e "\033[0;32m$*\033[0m"; }
yellow() { echo -e "\033[0;33m$*\033[0m"; }

die() { red "ERROR: $*"; exit 1; }

# ── Locate and extract model bundle ──────────────────────────────────────────

MODEL_BUNDLE_ZIP="$VLM_DIR/qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_gen5.zip"
MODEL_BUNDLE_DIR="$VLM_DIR/qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_gen5"

if [ ! -d "$MODEL_BUNDLE_DIR" ]; then
    if [ -f "$MODEL_BUNDLE_ZIP" ]; then
        yellow "Model bundle directory not found. Extracting zip..."
        unzip -q "$MODEL_BUNDLE_ZIP" -d "$VLM_DIR"
        green "Extracted: $MODEL_BUNDLE_DIR"
    else
        die "Model bundle not found. Expected:
  $MODEL_BUNDLE_DIR
  or
  $MODEL_BUNDLE_ZIP

Download from Qualcomm AI Hub:
  https://aihub.qualcomm.com/mobile/models/qwen3_vl_4b_instruct
  (Quick Start → OS: Android → Target: Snapdragon 8 Elite Gen 5 → geniex_qairt-w4a16)"
    fi
fi

# ── Read QAIRT version from model metadata ────────────────────────────────────

METADATA_FILE="$MODEL_BUNDLE_DIR/metadata.json"
if [ -f "$METADATA_FILE" ]; then
    REQUIRED_QAIRT=$(python3 -c \
        "import json,sys; d=json.load(open('$METADATA_FILE')); print(d['tool_versions']['qairt'])" 2>/dev/null || true)
fi

echo ""
if [ -n "$REQUIRED_QAIRT" ]; then
    echo "  Model bundle requires QAIRT SDK version: $REQUIRED_QAIRT"
    echo "  Download: https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK"
    echo ""
fi

# ── Validate QAIRT_SDK_ROOT ───────────────────────────────────────────────────

if [ -z "$QAIRT_SDK_ROOT" ]; then
    die "QAIRT_SDK_ROOT is not set. Please export it before running this script.
  Example:
    export QAIRT_SDK_ROOT=/path/to/qairt/${REQUIRED_QAIRT:-<version>}"
fi

if [ ! -d "$QAIRT_SDK_ROOT" ]; then
    die "QAIRT_SDK_ROOT directory does not exist: $QAIRT_SDK_ROOT"
fi

echo "  Using QAIRT SDK: $QAIRT_SDK_ROOT"

# Warn if SDK version doesn't match the model's required version
if [ -n "$REQUIRED_QAIRT" ]; then
    SDK_VERSION=$(basename "$QAIRT_SDK_ROOT")
    if [[ "$SDK_VERSION" != *"$REQUIRED_QAIRT"* ]]; then
        yellow "  Warning: SDK path '$SDK_VERSION' may not match required version '$REQUIRED_QAIRT'."
        yellow "           Mismatched QAIRT versions can cause load-time or runtime failures."
    fi
fi

# ── Create output directories ─────────────────────────────────────────────────

JNILIBS_DIR="$VLM_DIR/app/src/main/jniLibs/arm64-v8a"
ASSETS_DIR="$VLM_DIR/app/src/main/assets"
GENIE_HEADERS_DIR="$VLM_DIR/app/src/main/cpp/genie"
GENIE_BUNDLE_DIR="$VLM_DIR/genie_bundle"

mkdir -p "$JNILIBS_DIR"
mkdir -p "$ASSETS_DIR"
mkdir -p "$GENIE_BUNDLE_DIR"

# ── Helper: copy with existence check ────────────────────────────────────────

copy_file() {
    local src="$1"
    local dst_dir="$2"
    if [ -f "$src" ]; then
        cp "$src" "$dst_dir/"
        echo "  Copied: $(basename $src)"
    else
        yellow "  Warning: not found — $src"
    fi
}

# ── Helper: copy JSON and rewrite relative paths → absolute device paths ──────
#
# Model bundle JSONs use bare filenames (e.g. "vision_encoder.bin").
# Both the APK assets and the on-device genie_bundle require full paths
# (/data/local/tmp/genie_bundle/<file>) so Genie can locate them at runtime.

rewrite_json() {
    local src="$1"
    local dst="$2"
    python3 - "$src" "$dst" "$DEVICE_BUNDLE_PATH" <<'PYEOF'
import json, sys, re

src, dst, bundle = sys.argv[1], sys.argv[2], sys.argv[3]

with open(src) as f:
    text = f.read()

# Files that live in the on-device bundle directory
BUNDLE_FILES = [
    "vision_encoder.bin",
    "part1_of_4.bin", "part2_of_4.bin", "part3_of_4.bin", "part4_of_4.bin",
    "embedding_weights.raw",
    "tokenizer.json",
    "htp_backend_ext_config.json",
]

for fname in BUNDLE_FILES:
    # Match the bare filename in a JSON string value (not already prefixed)
    pattern = r'(?<![/\w])' + re.escape(fname) + r'(?![/\w])'
    replacement = bundle + "/" + fname
    text = re.sub(pattern, replacement, text)

with open(dst, "w") as f:
    f.write(text)

print(f"  Rewritten: {dst.split('/')[-1]}")
PYEOF
}

# ── Step 1: Copy Genie C API headers from QAIRT SDK ──────────────────────────

echo ""
echo "── Step 1: Copying Genie C API headers ──────────────────────────────────"

GENIE_SDK_HEADERS="$QAIRT_SDK_ROOT/include/Genie"
if [ -d "$GENIE_SDK_HEADERS" ]; then
    mkdir -p "$GENIE_HEADERS_DIR"
    echo "  Created: $GENIE_HEADERS_DIR"
    cp "$GENIE_SDK_HEADERS"/*.h "$GENIE_HEADERS_DIR/"
    echo "  Copied: $(ls "$GENIE_SDK_HEADERS"/*.h | wc -l | tr -d ' ') headers → $GENIE_HEADERS_DIR"
else
    die "Genie headers not found at: $GENIE_SDK_HEADERS
  Expected layout: \$QAIRT_SDK_ROOT/include/Genie/GenieNode.h ..."
fi

green "  Genie headers up to date."

# ── Step 2: Copy QAIRT runtime libraries to jniLibs ──────────────────────────

echo ""
echo "── Step 2: Copying QAIRT runtime libraries ──────────────────────────────"

# Core Genie + QNN HTP libraries (bundled in APK)
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libGenie.so"              "$JNILIBS_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtp.so"             "$JNILIBS_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so"      "$JNILIBS_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtpNetRunExtensions.so" "$JNILIBS_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnSystem.so"          "$JNILIBS_DIR"

# HTP v81 stubs — Snapdragon 8 Elite Gen 5 (DSP arch v81)
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so"       "$JNILIBS_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"  "$JNILIBS_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81.so"      "$JNILIBS_DIR"

# HTP v79 stubs — Snapdragon 8 Elite (DSP arch v79)
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtpV79Stub.so"       "$JNILIBS_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so"  "$JNILIBS_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/hexagon-v79/unsigned/libQnnHtpV79.so"      "$JNILIBS_DIR"

green "  jniLibs populated."

# ── Step 3: Copy and rewrite model config assets (go into APK) ───────────────

echo ""
echo "── Step 3: Copying model config assets (APK) ────────────────────────────"

# JSON configs — paths rewritten to absolute device paths
rewrite_json "$MODEL_BUNDLE_DIR/img-enc-htp.json"      "$ASSETS_DIR/img-enc-htp.json"
rewrite_json "$MODEL_BUNDLE_DIR/text-encoder.json"     "$ASSETS_DIR/text-encoder.json"
rewrite_json "$MODEL_BUNDLE_DIR/text-generator.json"   "$ASSETS_DIR/text-generator.json"
rewrite_json "$MODEL_BUNDLE_DIR/htp_backend_ext_config.json" "$ASSETS_DIR/htp_backend_ext_config.json"
rewrite_json "$MODEL_BUNDLE_DIR/genie_config.json"     "$ASSETS_DIR/genie_config.json"

# Tokenizer (referenced by absolute path in JSON; also keep a copy in assets)
copy_file "$MODEL_BUNDLE_DIR/tokenizer.json"          "$ASSETS_DIR"
copy_file "$MODEL_BUNDLE_DIR/tokenizer_config.json"   "$ASSETS_DIR"

# Fixed vision encoder tensors (precomputed for 512×512 input, never change)
copy_file "$MODEL_BUNDLE_DIR/sample_inputs/position_ids_cos.raw"      "$ASSETS_DIR"
copy_file "$MODEL_BUNDLE_DIR/sample_inputs/position_ids_sin.raw"      "$ASSETS_DIR"
copy_file "$MODEL_BUNDLE_DIR/sample_inputs/window_attention_mask.raw" "$ASSETS_DIR"
copy_file "$MODEL_BUNDLE_DIR/sample_inputs/full_attention_mask.raw"   "$ASSETS_DIR"

green "  Assets populated."

# ── Step 4: Populate genie_bundle (pushed to device, not in APK) ─────────────

echo ""
echo "── Step 4: Populating genie_bundle for device deployment ────────────────"

# JSON configs — same path rewriting as for assets
rewrite_json "$MODEL_BUNDLE_DIR/img-enc-htp.json"      "$GENIE_BUNDLE_DIR/img-enc-htp.json"
rewrite_json "$MODEL_BUNDLE_DIR/text-encoder.json"     "$GENIE_BUNDLE_DIR/text-encoder.json"
rewrite_json "$MODEL_BUNDLE_DIR/text-generator.json"   "$GENIE_BUNDLE_DIR/text-generator.json"
rewrite_json "$MODEL_BUNDLE_DIR/htp_backend_ext_config.json" "$GENIE_BUNDLE_DIR/htp_backend_ext_config.json"
rewrite_json "$MODEL_BUNDLE_DIR/genie_config.json"     "$GENIE_BUNDLE_DIR/genie_config.json"

copy_file "$MODEL_BUNDLE_DIR/tokenizer.json"           "$GENIE_BUNDLE_DIR"

# Heavy model binaries (NOT in APK — too large; pushed separately via adb)
echo ""
echo "  Copying model binaries (this may take a moment)..."
copy_file "$MODEL_BUNDLE_DIR/vision_encoder.bin"   "$GENIE_BUNDLE_DIR"
copy_file "$MODEL_BUNDLE_DIR/part1_of_4.bin"       "$GENIE_BUNDLE_DIR"
copy_file "$MODEL_BUNDLE_DIR/part2_of_4.bin"       "$GENIE_BUNDLE_DIR"
copy_file "$MODEL_BUNDLE_DIR/part3_of_4.bin"       "$GENIE_BUNDLE_DIR"
copy_file "$MODEL_BUNDLE_DIR/part4_of_4.bin"       "$GENIE_BUNDLE_DIR"
copy_file "$MODEL_BUNDLE_DIR/embedding_weights.raw" "$GENIE_BUNDLE_DIR"

# QAIRT runtime libs needed on-device (loaded via ADSP_LIBRARY_PATH)
copy_file "$QAIRT_SDK_ROOT/bin/aarch64-android/genie-t2t-run"              "$GENIE_BUNDLE_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libGenie.so"                "$GENIE_BUNDLE_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtp.so"               "$GENIE_BUNDLE_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtpPrepare.so"        "$GENIE_BUNDLE_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnSystem.so"            "$GENIE_BUNDLE_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtpNetRunExtensions.so" "$GENIE_BUNDLE_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/aarch64-android/libQnnHtpV81Stub.so"        "$GENIE_BUNDLE_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"   "$GENIE_BUNDLE_DIR"
copy_file "$QAIRT_SDK_ROOT/lib/hexagon-v81/unsigned/libQnnHtpV81.so"       "$GENIE_BUNDLE_DIR"

green "  genie_bundle populated."

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
green "════════════════════════════════════════════════════════════════════════"
green " Dependencies resolved successfully!"
green "════════════════════════════════════════════════════════════════════════"
echo ""
echo " Next steps:"
echo "   1. Push model bundle to device:"
echo "        adb push genie_bundle/ $DEVICE_BUNDLE_PATH/"
echo "   2. Open project in Android Studio and build/install the APK"
echo ""
