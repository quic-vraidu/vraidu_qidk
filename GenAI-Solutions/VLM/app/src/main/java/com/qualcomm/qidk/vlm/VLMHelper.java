//============================================================================
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//============================================================================

package com.qualcomm.qidk.vlm;

import android.content.Context;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * Config loading and Qwen2.5-VL prompt formatting utilities.
 *
 * <p>VLM prompt structure (from sample_inputs/):
 * <pre>
 *   PREFIX:
 *     {@code <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>}
 *   [IMAGE EMBEDDINGS injected by imageEncoder + lutEncoder pipeline]
 *   SUFFIX:
 *     {@code <|vision_end|>{user_text}<|im_end|>\n<|im_start|>assistant\n}
 *
 *   Text-only prompt:
 *     {@code <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n}
 * </pre>
 */
public class VLMHelper {

    private static final String TAG = "VLMHelper";

    // Qwen2.5-VL fixed prompt tokens
    private static final String IM_START     = "<|im_start|>";
    private static final String IM_END       = "<|im_end|>";
    private static final String VISION_START = "<|vision_start|>";
    private static final String VISION_END   = "<|vision_end|>";
    private static final String SYSTEM_TEXT  = "You are a helpful assistant.";

    // -------------------------------------------------------------------------
    // Prompt builders
    // -------------------------------------------------------------------------

    /**
     * Returns the fixed prefix injected before image embeddings.
     * Ends just before the image embeddings.
     */
    public static String buildVLMPrefix() {
        return IM_START + "system\n" + SYSTEM_TEXT + IM_END + "\n"
             + IM_START + "user\n"
             + VISION_START;
    }

    /**
     * Returns the suffix injected after image embeddings, incorporating the user's text.
     */
    public static String buildVLMSuffix(String userText) {
        return VISION_END + userText + IM_END + "\n" + IM_START + "assistant\n";
    }

    /**
     * Text-only (no image) prompt – sent as a single string to lutEncoder.
     */
    public static String buildTextPrompt(String userText) {
        return IM_START + "system\n" + SYSTEM_TEXT + IM_END + "\n"
             + IM_START + "user\n" + userText + IM_END + "\n"
             + IM_START + "assistant\n";
    }

    // -------------------------------------------------------------------------
    // Asset loading
    // -------------------------------------------------------------------------

    /** Reads a text or binary asset as a UTF-8 String. */
    public static String loadTextAsset(Context context, String fileName) {
        try (InputStream is = context.getAssets().open(fileName)) {
            byte[] buf = new byte[is.available()];
            //noinspection ResultOfMethodCallIgnored
            is.read(buf);
            return new String(buf, StandardCharsets.UTF_8);
        } catch (IOException e) {
            Log.e(TAG, "loadTextAsset failed: " + fileName, e);
            return null;
        }
    }

    // -------------------------------------------------------------------------
    // URI helpers
    // -------------------------------------------------------------------------

    /** Extracts the file name from a content URI (best-effort). */
    public static String getFileName(android.net.Uri uri) {
        if (uri == null) return null;
        String path = uri.getLastPathSegment();
        if (path != null && path.contains("/")) {
            return path.substring(path.lastIndexOf('/') + 1);
        }
        return path;
    }
}
