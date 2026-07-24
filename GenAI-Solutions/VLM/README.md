## VLM — Vision-Language Model powered by 

This VLM app enables on-device multimodal AI — attach an image, ask a question, and get a streamed answer entirely on the Snapdragon NPU. It demonstrates how to use the Genie C APIs from [QAIRT SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) to run a three-node Vision-Language Model pipeline (image encoder + text encoder + LLM decoder) on the Hexagon DSP.

## Requirements

### Platform

- Snapdragon® 8 Elite Gen 5 device (e.g. QIDK devkit or flagship phone with SM8850)

### Model

- [Qwen3-VL](https://aihub.qualcomm.com/mobile/models/qwen3_vl_4b_instruct)

### Tools and SDK

- Download [Android Studio](https://developer.android.com/studio/archive). **App is tested with Android Studio Panda 4 Version 2025.3.4.**
- Download and extract [Qualcomm® AI Runtime SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) for Linux.  
  Set the environment variable:
  ```bash
  export QAIRT_SDK_ROOT=<path to extracted QAIRT SDK>
  ```

### Download the VLM Model from AI Hub

The app uses **Qwen3-VL-4B-Instruct** quantized to w4a16 for the Snapdragon 8 Elite Gen 5 HTP.

1. Download the pre-compiled model bundle from [Qualcomm AI Hub](https://aihub.qualcomm.com/models/qwen3_vl_4b_instruct):
   - Quick Start: Select OS: Android
   - Select target: **Qualcomm Snapdragon 8 Elite Gen 5**
   - Download the `geniex_qairt-w4a16` variant

2. Place the downloaded zip (or extracted directory) in the `VLM/` project root:
   ```
   GenAI-Solutions/VLM/
   └── qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_gen5.zip
   ```
   The `resolveDependencies.sh` script will extract it automatically if needed.

   > **Note:** Exporting the model via AI Hub requires a Qualcomm AI Hub account. Sign up at [aihub.qualcomm.com](https://aihub.qualcomm.com).

3. Check the required QAIRT SDK version from `metadata.json` inside the downloaded bundle:
   ```bash
   python3 -c "import json; d=json.load(open('qwen3_vl_4b_instruct-geniex_qairt-w4a16-qualcomm_snapdragon_8_elite_gen5/metadata.json')); print(d['tool_versions']['qairt'])"
   ```
   Example output: `2.45.0.260326154327`

   Download the matching [Qualcomm® AI Runtime SDK](https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK) version and extract it.

   > Having mismatched QAIRT versions can cause load-time or runtime failures. The `resolveDependencies.sh` script will automatically read the version from `metadata.json` and warn you if the SDK path looks incorrect.

## Build App

1. Go to the VLM directory:
   ```bash
   cd <qidk path>/GenAI-Solutions/VLM/
   ```

2. Set the QAIRT SDK path and run the dependency script:
   ```bash
   export QAIRT_SDK_ROOT=<path to QAIRT version used by AI Hub>
   source scripts/resolveDependencies.sh
   ```

   This script will:
   - Read the required QAIRT version from `metadata.json` and warn on mismatch
   - Copy Genie C API headers from `$QAIRT_SDK_ROOT/include/Genie/` into `app/src/main/cpp/genie/`
   - Copy QAIRT runtime libraries (`.so` files) into `app/src/main/jniLibs/arm64-v8a/`
   - Copy and rewrite JSON configs (relative → absolute device paths), tokenizer, and fixed tensor assets into `app/src/main/assets/`
   - Populate `genie_bundle/` with model binaries, rewritten configs, and runtime libs for device deployment

3. Push the model bundle to the device:
   ```bash
   adb push genie_bundle/ /data/local/tmp/genie_bundle/
   ```

4. Open the project in Android Studio, build, and install the APK on the device.

## Demo — VLM App Usage

The app supports two modes:

| Mode | How to use |
|---|---|
| **VLM (Image + Text)** | Tap the image icon, select a photo, type a question, press Send |
| **Text-only** | Type a question directly and press Send (no image needed) |

Responses stream token-by-token in real time directly on the Snapdragon NPU.

<img src="./demo/VLM.gif" width="360" height="800">