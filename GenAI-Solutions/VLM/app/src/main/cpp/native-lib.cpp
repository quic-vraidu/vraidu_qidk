//============================================================================
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//============================================================================

#include <jni.h>
#include <string>
#include <cstring>
#include <cctype>
#include <time.h>
#include <android/log.h>
#include <genie/GenieCommon.h>
#include <genie/GenieNode.h>
#include <genie/GeniePipeline.h>

#define LOG_TAG  "VLMAssistant_JNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)

// Expected tensor sizes for Qwen2.5-VL (336×504, patch 14×14, temporal 2)
static const size_t EXPECTED_PIXEL_VALUES_BYTES = 1728 * 588 * sizeof(float); // ~3.9 MB
static const size_t EXPECTED_POS_COS_BYTES      = 1728 * 20  * sizeof(float); // 138 240 B
static const size_t EXPECTED_POS_SIN_BYTES      = 1728 * 20  * sizeof(float);
static const size_t EXPECTED_WINDOW_MASK_BYTES  = 864  * 864 * sizeof(float); // ~3.0 MB
static const size_t EXPECTED_FULL_MASK_BYTES    = 864  * 864 * sizeof(float);

// Per-inference token counter (reset at start of each inference)
static int  g_tokenCount    = 0;
static bool g_seenNonPrint  = false;

static inline long long nowMs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
}

static const char* sentenceCodeName(GenieNode_TextOutput_SentenceCode_t code) {
    switch (code) {
        case GENIE_NODE_SENTENCE_COMPLETE: return "COMPLETE";
        case GENIE_NODE_SENTENCE_BEGIN:    return "BEGIN";
        case GENIE_NODE_SENTENCE_CONTINUE: return "CONTINUE";
        case GENIE_NODE_SENTENCE_END:      return "END";
        case GENIE_NODE_SENTENCE_ABORT:    return "ABORT";
        case GENIE_NODE_SENTENCE_REWIND:   return "REWIND";
        default:                           return "UNKNOWN";
    }
}

// Returns true if every byte in str is printable ASCII or common whitespace
static bool isPrintableUtf8(const char* str) {
    if (!str) return true;
    for (const unsigned char* p = (const unsigned char*)str; *p; ++p) {
        // Allow printable ASCII and common control chars (tab, LF, CR)
        if (*p < 0x09 || (*p > 0x0D && *p < 0x20 && *p != 0x1B)) return false;
    }
    return true;
}

// Log the first N floats from a byte buffer (for sanity-checking tensor content)
static void logFirstFloats(const char* label, const jbyte* buf, jsize bytes, int n = 8) {
    if (!buf || bytes < (jsize)(n * sizeof(float))) return;
    const float* fp = reinterpret_cast<const float*>(buf);
    char tmp[256];
    int off = 0;
    for (int i = 0; i < n && off < 240; ++i)
        off += snprintf(tmp + off, sizeof(tmp) - off, "%.4f ", fp[i]);
    LOGD("%s first %d floats: %s", label, n, tmp);
}
#define CHECK(s, msg) if ((s) != GENIE_STATUS_SUCCESS) { LOGE(msg ": %d", (int)(s)); return (s); }

// On-device bundle path (all .bin files must be here)
static const char* GENIE_BUNDLE_PATH = "/data/local/tmp/genie_bundle/";

// ─── Global pipeline handles ───────────────────────────────────────────────
static GeniePipelineConfig_Handle_t g_pipelineConfig   = nullptr;
static GeniePipeline_Handle_t       g_pipeline          = nullptr;
static GenieNodeConfig_Handle_t     g_imgEncConfig      = nullptr;
static GenieNodeConfig_Handle_t     g_lutEncConfig      = nullptr;
static GenieNodeConfig_Handle_t     g_textGenConfig     = nullptr;
static GenieNode_Handle_t           g_imageEncoderNode  = nullptr;
static GenieNode_Handle_t           g_lutEncoderNode    = nullptr;
static GenieNode_Handle_t           g_textGeneratorNode = nullptr;

// ─── Streaming callback state (set per inference call) ────────────────────
static JNIEnv*   g_jniEnv        = nullptr;
static jobject   g_tokenCallback  = nullptr;
static jmethodID g_onTokenMethod  = nullptr;

//─────────────────────────────────────────────────────────────────────────────
// Helper: set ADSP_LIBRARY_PATH so DSP stubs and model .bin files are found
//─────────────────────────────────────────────────────────────────────────────
static bool setAdspPaths(const std::string& nativeLibPath) {
    std::string path = nativeLibPath + ";" + GENIE_BUNDLE_PATH;
    LOGI("ADSP_LIBRARY_PATH = %s", path.c_str());
    return (setenv("ADSP_LIBRARY_PATH", path.c_str(), 1) == 0 &&
            setenv("LD_LIBRARY_PATH",   path.c_str(), 1) == 0);
}

//─────────────────────────────────────────────────────────────────────────────
// Helper: free all pipeline + node resources
//─────────────────────────────────────────────────────────────────────────────
static void freePipelineResources() {
    if (g_pipeline)          { GeniePipeline_free(g_pipeline);          g_pipeline          = nullptr; }
    if (g_pipelineConfig)    { GeniePipelineConfig_free(g_pipelineConfig); g_pipelineConfig  = nullptr; }
    if (g_imageEncoderNode)  { GenieNode_free(g_imageEncoderNode);       g_imageEncoderNode  = nullptr; }
    if (g_lutEncoderNode)    { GenieNode_free(g_lutEncoderNode);         g_lutEncoderNode    = nullptr; }
    if (g_textGeneratorNode) { GenieNode_free(g_textGeneratorNode);      g_textGeneratorNode = nullptr; }
    if (g_imgEncConfig)      { GenieNodeConfig_free(g_imgEncConfig);     g_imgEncConfig      = nullptr; }
    if (g_lutEncConfig)      { GenieNodeConfig_free(g_lutEncConfig);     g_lutEncConfig      = nullptr; }
    if (g_textGenConfig)     { GenieNodeConfig_free(g_textGenConfig);    g_textGenConfig     = nullptr; }
}

//═════════════════════════════════════════════════════════════════════════════
// JNI: loadPipeline
//   imgEncJson   – contents of img-enc-htp.json
//   lutEncJson   – contents of text-encoder.json
//   textGenJson  – contents of text-generator.json
//   nativeLibPath – app's nativeLibraryDir
// Returns 0 on success.
//═════════════════════════════════════════════════════════════════════════════
extern "C"
JNIEXPORT jint JNICALL
Java_com_qualcomm_qidk_vlm_MainActivity_loadPipeline(
        JNIEnv* env, jobject,
        jstring imgEncJson, jstring lutEncJson, jstring textGenJson,
        jstring nativeLibPath) {

    freePipelineResources();

    const char* imgEncStr   = env->GetStringUTFChars(imgEncJson,   nullptr);
    const char* lutEncStr   = env->GetStringUTFChars(lutEncJson,   nullptr);
    const char* textGenStr  = env->GetStringUTFChars(textGenJson,  nullptr);
    const char* nativeLib   = env->GetStringUTFChars(nativeLibPath, nullptr);
    std::string nativeLibStr(nativeLib);
    env->ReleaseStringUTFChars(nativeLibPath, nativeLib);

    // Step 1: Set DSP paths so .so and .bin files are found
    if (!setAdspPaths(nativeLibStr)) {
        LOGE("setAdspPaths failed");
        env->ReleaseStringUTFChars(imgEncJson,  imgEncStr);
        env->ReleaseStringUTFChars(lutEncJson,  lutEncStr);
        env->ReleaseStringUTFChars(textGenJson, textGenStr);
        return -1;
    }

    Genie_Status_t s;

    // Helper to print long strings in 512-byte chunks (logcat line limit)
    auto logFull = [](const char* label, const char* str) {
        const int CHUNK = 512;
        int len = (int)strlen(str);
        for (int i = 0; i < len; i += CHUNK) {
            __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "%s[%d]: %.*s",
                                label, i, CHUNK, str + i);
        }
    };

    // Step 2: Create node configs from JSON
    LOGI("loading imageEncoder JSON: img-enc-htp.json");
    //logFull("img-enc-htp.json", imgEncStr);
    s = GenieNodeConfig_createFromJson(imgEncStr,  &g_imgEncConfig);
    env->ReleaseStringUTFChars(imgEncJson, imgEncStr);
    CHECK(s, "GenieNodeConfig_createFromJson(imageEncoder)");
    LOGI("imageEncoder config: OK");

    LOGI("loading lutEncoder JSON: text-encoder.json");
    //logFull("text-encoder.json", lutEncStr);
    s = GenieNodeConfig_createFromJson(lutEncStr,  &g_lutEncConfig);
    env->ReleaseStringUTFChars(lutEncJson, lutEncStr);
    CHECK(s, "GenieNodeConfig_createFromJson(lutEncoder)");
    LOGI("lutEncoder config: OK");

    LOGI("loading textGenerator JSON: text-generator.json");
    //logFull("text-generator.json", textGenStr);
    s = GenieNodeConfig_createFromJson(textGenStr, &g_textGenConfig);
    env->ReleaseStringUTFChars(textGenJson, textGenStr);
    CHECK(s, "GenieNodeConfig_createFromJson(textGenerator)");
    LOGI("textGenerator config: OK");

    // Step 3: Create nodes (this loads model weights into HTP memory)
    s = GenieNode_create(g_imgEncConfig,  &g_imageEncoderNode);
    CHECK(s, "GenieNode_create(imageEncoder)");
    LOGI("imageEncoder node: OK");

    s = GenieNode_create(g_lutEncConfig,  &g_lutEncoderNode);
    CHECK(s, "GenieNode_create(lutEncoder)");
    LOGI("lutEncoder node: OK");

    s = GenieNode_create(g_textGenConfig, &g_textGeneratorNode);
    CHECK(s, "GenieNode_create(textGenerator)");
    LOGI("textGenerator node: OK");

    // Step 4: Register streaming text callback on textGenerator
    GenieNode_TextOutput_Callback_t textCb = [](
            const char* response,
            const GenieNode_TextOutput_SentenceCode_t sentenceCode,
            const void* /*userData*/) -> Genie_Status_t {

        ++g_tokenCount;

        // Log sentence events that mark boundaries
        if (sentenceCode != GENIE_NODE_SENTENCE_CONTINUE) {
            LOGI("[token #%d] sentenceCode=%s response=\"%s\"",
                 g_tokenCount, sentenceCodeName(sentenceCode),
                 response ? response : "(null)");
        } else {
            LOGD("[token #%d] \"%s\"", g_tokenCount, response ? response : "(null)");
        }

        // Warn once if we see non-printable / binary content in a token
        if (!g_seenNonPrint && response && !isPrintableUtf8(response)) {
            g_seenNonPrint = true;
            LOGW("[token #%d] NON-PRINTABLE bytes detected – raw hex:", g_tokenCount);
            const unsigned char* p = (const unsigned char*)response;
            char hex[128]; int off = 0;
            for (int i = 0; i < 32 && p[i] && off < 120; ++i)
                off += snprintf(hex + off, sizeof(hex) - off, "%02X ", p[i]);
            LOGW("  hex dump: %s", hex);
        }

        if (sentenceCode == GENIE_NODE_SENTENCE_ABORT) {
            LOGE("Inference ABORTED by model after %d tokens", g_tokenCount);
        }

        if (!g_jniEnv || !g_tokenCallback || !g_onTokenMethod) return GENIE_STATUS_SUCCESS;
        jstring jToken = g_jniEnv->NewStringUTF(response ? response : "");
        g_jniEnv->CallVoidMethod(g_tokenCallback, g_onTokenMethod, jToken);
        g_jniEnv->DeleteLocalRef(jToken);
        return GENIE_STATUS_SUCCESS;
    };

    s = GenieNode_setTextCallback(g_textGeneratorNode,
                                  GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT,
                                  textCb);
    CHECK(s, "GenieNode_setTextCallback");
    LOGI("textCallback registered: OK");

    // Step 5: Create pipeline config and pipeline
    s = GeniePipelineConfig_createFromJson("{\"version\": 1}", &g_pipelineConfig);
    CHECK(s, "GeniePipelineConfig_createFromJson");

    s = GeniePipeline_create(g_pipelineConfig, &g_pipeline);
    CHECK(s, "GeniePipeline_create");
    LOGI("pipeline created: OK");

    // Step 6: Add nodes to pipeline
    s = GeniePipeline_addNode(g_pipeline, g_imageEncoderNode);
    CHECK(s, "GeniePipeline_addNode(imageEncoder)");

    s = GeniePipeline_addNode(g_pipeline, g_lutEncoderNode);
    CHECK(s, "GeniePipeline_addNode(lutEncoder)");

    s = GeniePipeline_addNode(g_pipeline, g_textGeneratorNode);
    CHECK(s, "GeniePipeline_addNode(textGenerator)");

    // Step 7: Connect image encoder → text generator
    s = GeniePipeline_connect(g_pipeline,
                              g_imageEncoderNode,  GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT,
                              g_textGeneratorNode, GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT);
    CHECK(s, "GeniePipeline_connect(imageEncoder → textGenerator)");

    // Step 8: Connect lut encoder → text generator
    s = GeniePipeline_connect(g_pipeline,
                              g_lutEncoderNode,    GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT,
                              g_textGeneratorNode, GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT);
    CHECK(s, "GeniePipeline_connect(lutEncoder → textGenerator)");

    LOGI("VLM pipeline fully loaded and connected");
    return 0;
}

//═════════════════════════════════════════════════════════════════════════════
// JNI: runVLMInference  – image + text
//
//   prefixText  – prompt before image (e.g. <|im_start|>system...<|vision_start|>)
//   pixelValues – float32 patchified image [1728 × 588] as byte[]
//   posCos      – float32 RoPE cos [1728 × 20] as byte[]
//   posSin      – float32 RoPE sin [1728 × 20] as byte[]
//   windowMask  – float32 window attention mask [864 × 864] as byte[]
//   fullMask    – float32 full attention mask [864 × 864] as byte[]
//   suffixText  – prompt after image (e.g. <|vision_end|>...assistant\n)
//   callback    – Java TokenCallback.onToken(String)
//
// Returns 0 on success.
//═════════════════════════════════════════════════════════════════════════════
extern "C"
JNIEXPORT jint JNICALL
Java_com_qualcomm_qidk_vlm_VLMChatActivity_runVLMInference(
        JNIEnv* env, jobject,
        jstring  prefixText,
        jbyteArray pixelValues,
        jbyteArray posCos,
        jbyteArray posSin,
        jbyteArray windowMask,
        jbyteArray fullMask,
        jstring  suffixText,
        jobject  callback) {

    if (!g_pipeline) { LOGE("Pipeline not loaded"); return -1; }

    // Reset per-inference counters
    g_tokenCount   = 0;
    g_seenNonPrint = false;

    // Store JNI state for streaming callback
    g_jniEnv        = env;
    g_tokenCallback = callback;
    jclass cls = env->GetObjectClass(callback);
    g_onTokenMethod = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");

    Genie_Status_t s;

    // ── 0. Log and validate inputs ──────────────────────────────────────────
    const char* prefixDbg = env->GetStringUTFChars(prefixText, nullptr);
    const char* suffixDbg = env->GetStringUTFChars(suffixText, nullptr);
    LOGI("runVLMInference: prefix_len=%zu suffix_len=%zu",
         strlen(prefixDbg), strlen(suffixDbg));
    LOGD("  prefix: \"%.120s\"", prefixDbg);
    LOGD("  suffix: \"%.120s\"", suffixDbg);
    env->ReleaseStringUTFChars(prefixText, prefixDbg);
    env->ReleaseStringUTFChars(suffixText, suffixDbg);

    jsize szPixel  = env->GetArrayLength(pixelValues);
    jsize szCos    = env->GetArrayLength(posCos);
    jsize szSin    = env->GetArrayLength(posSin);
    jsize szWin    = env->GetArrayLength(windowMask);
    jsize szFull   = env->GetArrayLength(fullMask);
    LOGI("  tensor sizes – pixelValues:%d  posCos:%d  posSin:%d  windowMask:%d  fullMask:%d",
         szPixel, szCos, szSin, szWin, szFull);
    if ((size_t)szPixel != EXPECTED_PIXEL_VALUES_BYTES)
        LOGW("  pixelValues size mismatch: expected %zu got %d", EXPECTED_PIXEL_VALUES_BYTES, szPixel);
    if ((size_t)szCos != EXPECTED_POS_COS_BYTES)
        LOGW("  posCos size mismatch: expected %zu got %d", EXPECTED_POS_COS_BYTES, szCos);
    if ((size_t)szSin != EXPECTED_POS_SIN_BYTES)
        LOGW("  posSin size mismatch: expected %zu got %d", EXPECTED_POS_SIN_BYTES, szSin);
    if ((size_t)szWin != EXPECTED_WINDOW_MASK_BYTES)
        LOGW("  windowMask size mismatch: expected %zu got %d", EXPECTED_WINDOW_MASK_BYTES, szWin);
    if ((size_t)szFull != EXPECTED_FULL_MASK_BYTES)
        LOGW("  fullMask size mismatch: expected %zu got %d", EXPECTED_FULL_MASK_BYTES, szFull);

    // Log first few floats from pixelValues to verify CHW layout.
    // With CHW, float[0..195]=R, float[196..391]=G, float[392..587]=B for patch 0.
    // Show first 4 R, first 4 G, first 4 B values to confirm channel separation.
    {
        jbyte* pvBuf = env->GetByteArrayElements(pixelValues, nullptr);
        if (pvBuf && szPixel >= (jsize)(588 * sizeof(float))) {
            const float* fp = reinterpret_cast<const float*>(pvBuf);
            LOGD("pixelValues patch[0] CHW layout check:"
                 " R[0..3]=%.3f %.3f %.3f %.3f"
                 " G[0..3]=%.3f %.3f %.3f %.3f"
                 " B[0..3]=%.3f %.3f %.3f %.3f",
                 fp[0], fp[1], fp[2], fp[3],
                 fp[196], fp[197], fp[198], fp[199],
                 fp[392], fp[393], fp[394], fp[395]);
        }
        env->ReleaseByteArrayElements(pixelValues, pvBuf, JNI_ABORT);
    }

    // ── 1. Set prompt prefix on lutEncoder ──────────────────────────────────
    const char* prefix = env->GetStringUTFChars(prefixText, nullptr);
    LOGD("setData prefix (%zu bytes) on lutEncoder", strlen(prefix));
    s = GenieNode_setData(g_lutEncoderNode,
                          GENIE_NODE_TEXT_ENCODER_TEXT_INPUT,
                          prefix, strlen(prefix), nullptr);
    env->ReleaseStringUTFChars(prefixText, prefix);
    CHECK(s, "GenieNode_setData(prefix)");

    // ── 2. Set image data on imageEncoder ──────────────────────────────────
    auto setByteArrayData = [&](jbyteArray arr, GenieNode_IOName_t ioName, const char* label) -> Genie_Status_t {
        jbyte* buf  = env->GetByteArrayElements(arr, nullptr);
        jsize  size = env->GetArrayLength(arr);
        Genie_Status_t st = GenieNode_setData(g_imageEncoderNode, ioName,
                                              buf, (size_t)size, nullptr);
        env->ReleaseByteArrayElements(arr, buf, JNI_ABORT);
        if (st != GENIE_STATUS_SUCCESS) LOGE("GenieNode_setData(%s): %d", label, st);
        return st;
    };

    s = setByteArrayData(pixelValues, GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT,       "pixel_values");
    CHECK(s, "setData pixel_values");
    s = setByteArrayData(posCos,      GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS,     "pos_cos");
    CHECK(s, "setData pos_cos");
    s = setByteArrayData(posSin,      GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN,     "pos_sin");
    CHECK(s, "setData pos_sin");
    s = setByteArrayData(windowMask,  GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK, "window_mask");
    CHECK(s, "setData window_mask");
    s = setByteArrayData(fullMask,    GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK,   "full_mask");
    CHECK(s, "setData full_mask");

    // ── 3. Set prompt suffix on lutEncoder ──────────────────────────────────
    const char* suffix = env->GetStringUTFChars(suffixText, nullptr);
    LOGD("setData suffix (%zu bytes) on lutEncoder", strlen(suffix));
    s = GenieNode_setData(g_lutEncoderNode,
                          GENIE_NODE_TEXT_ENCODER_TEXT_INPUT,
                          suffix, strlen(suffix), nullptr);
    env->ReleaseStringUTFChars(suffixText, suffix);
    CHECK(s, "GenieNode_setData(suffix)");

    // ── 4. Execute pipeline (blocking, streams tokens via callback) ──────────
    LOGI("GeniePipeline_execute starting (VLM)...");
    long long t0 = nowMs();
    s = GeniePipeline_execute(g_pipeline, nullptr);
    long long elapsed = nowMs() - t0;
    if (s != GENIE_STATUS_SUCCESS && s != GENIE_STATUS_WARNING_CONTEXT_EXCEEDED) {
        LOGE("GeniePipeline_execute: %d (elapsed %lldms)", s, elapsed);
        return s;
    }
    LOGI("VLM inference complete: %d tokens in %lldms (status=%d)", g_tokenCount, elapsed, s);
    return 0;
}

//═════════════════════════════════════════════════════════════════════════════
// JNI: runTextInference  – text only (no image)
//
//   prompt   – full formatted Qwen2.5-VL prompt string
//   callback – Java TokenCallback.onToken(String)
//═════════════════════════════════════════════════════════════════════════════
extern "C"
JNIEXPORT jint JNICALL
Java_com_qualcomm_qidk_vlm_VLMChatActivity_runTextInference(
        JNIEnv* env, jobject,
        jstring prompt,
        jobject callback) {

    if (!g_pipeline) { LOGE("Pipeline not loaded"); return -1; }

    g_tokenCount   = 0;
    g_seenNonPrint = false;

    g_jniEnv        = env;
    g_tokenCallback = callback;
    jclass cls = env->GetObjectClass(callback);
    g_onTokenMethod = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");

    const char* promptStr = env->GetStringUTFChars(prompt, nullptr);
    LOGI("runTextInference: prompt_len=%zu", strlen(promptStr));
    LOGD("  prompt: \"%.200s\"", promptStr);
    Genie_Status_t s = GenieNode_setData(g_lutEncoderNode,
                                         GENIE_NODE_TEXT_ENCODER_TEXT_INPUT,
                                         promptStr, strlen(promptStr), nullptr);
    env->ReleaseStringUTFChars(prompt, promptStr);
    CHECK(s, "GenieNode_setData(text prompt)");

    LOGI("GeniePipeline_execute starting (text-only)...");
    long long t0 = nowMs();
    s = GeniePipeline_execute(g_pipeline, nullptr);
    long long elapsed = nowMs() - t0;
    if (s != GENIE_STATUS_SUCCESS && s != GENIE_STATUS_WARNING_CONTEXT_EXCEEDED) {
        LOGE("GeniePipeline_execute (text): %d (elapsed %lldms)", s, elapsed);
        return s;
    }
    LOGI("Text inference complete: %d tokens in %lldms (status=%d)", g_tokenCount, elapsed, s);
    return 0;
}

//═════════════════════════════════════════════════════════════════════════════
// JNI: resetPipeline – clears KV-cache between conversation turns
//═════════════════════════════════════════════════════════════════════════════
extern "C"
JNIEXPORT jint JNICALL
Java_com_qualcomm_qidk_vlm_VLMChatActivity_resetPipeline(JNIEnv*, jobject) {
    if (!g_pipeline) return -1;
    Genie_Status_t s = GeniePipeline_reset(g_pipeline);
    if (s == GENIE_STATUS_SUCCESS) LOGI("Pipeline reset: OK");
    else LOGE("Pipeline reset: %d", s);
    return s;
}

//═════════════════════════════════════════════════════════════════════════════
// JNI: freePipeline – release all Genie resources (called from onDestroy)
//═════════════════════════════════════════════════════════════════════════════
extern "C"
JNIEXPORT jint JNICALL
Java_com_qualcomm_qidk_vlm_MainActivity_freePipeline(JNIEnv*, jobject) {
    freePipelineResources();
    LOGI("Pipeline freed");
    return 0;
}
