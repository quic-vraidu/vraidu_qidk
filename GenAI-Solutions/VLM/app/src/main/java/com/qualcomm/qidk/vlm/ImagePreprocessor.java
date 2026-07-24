//============================================================================
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//============================================================================

package com.qualcomm.qidk.vlm;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Preprocesses an image into the float32 patchified format expected by the
 * Qwen3-VL-4B-Instruct vision encoder.
 *
 * <p>Model requirements (from img-enc-htp.json + metadata.json):
 * <ul>
 *   <li>Target image size: 512 × 512 pixels (square)</li>
 *   <li>Patch size: 16 × 16 pixels</li>
 *   <li>Temporal patch size: 2 (packed per spatial patch, not stacked separately)</li>
 *   <li>Grid: 32 rows × 32 cols = 1024 spatial patches</li>
 *   <li>Each patch: [T=2, C=3, H=16, W=16] = 1536 floats (both temporal copies packed inline)</li>
 *   <li>Output shape: [1024, 1536] → 1,572,864 floats → ~6 MB float32</li>
 * </ul>
 *
 * <p>Normalisation constants (Qwen3-VL):
 * <pre>
 *   mean = [0.5, 0.5, 0.5]
 *   std  = [0.5, 0.5, 0.5]
 *   normalised = (pixel / 255.0 - 0.5) / 0.5  =  pixel / 127.5 - 1.0   (range: [-1, +1])
 * </pre>
 */
public class ImagePreprocessor {

    private static final String TAG = "VLM_ImagePreprocessor";

    // Target image dimensions (square)
    public static final int TARGET_HEIGHT = 512;
    public static final int TARGET_WIDTH  = 512;

    // Patch dimensions
    private static final int PATCH_SIZE    = 16;
    private static final int PATCH_ROWS    = TARGET_HEIGHT / PATCH_SIZE;  // 32
    private static final int PATCH_COLS    = TARGET_WIDTH  / PATCH_SIZE;  // 32
    private static final int NUM_SPATIAL   = PATCH_ROWS * PATCH_COLS;     // 1024
    private static final int TEMPORAL_SIZE = 2;                            // packed per spatial patch
    private static final int NUM_PATCHES   = NUM_SPATIAL;                 // 1024 entries, each holds both temporal copies
    // Each patch entry: [T, C, H, W] = [2, 3, 16, 16] = 1536 floats
    private static final int PATCH_FLOATS  = TEMPORAL_SIZE * 3 * PATCH_SIZE * PATCH_SIZE; // 1536

    // Normalisation — Qwen3-VL uses simple [-1, +1] mapping
    private static final float[] MEAN = {0.5f, 0.5f, 0.5f};
    private static final float[] STD  = {0.5f, 0.5f, 0.5f};

    /**
     * Loads an image from a content URI, resizes to 336×504, and converts to
     * the float32 patchified byte array expected by the vision encoder.
     *
     * @param context Android context (for ContentResolver)
     * @param imageUri content:// URI of the selected image
     * @return byte[] containing float32 data [1728 × 588 floats], or null on error
     */
    public static byte[] preprocessFromUri(Context context, Uri imageUri) {
        try {
            // ── Query ContentResolver metadata before decoding ────────────────
            Log.i(TAG, "=== Input image metadata ===");
            Log.i(TAG, "  URI: " + imageUri);
            try (android.database.Cursor cursor = context.getContentResolver().query(
                    imageUri,
                    new String[]{
                        android.provider.OpenableColumns.DISPLAY_NAME,
                        android.provider.OpenableColumns.SIZE,
                        android.provider.MediaStore.Images.ImageColumns.MIME_TYPE,
                        android.provider.MediaStore.Images.ImageColumns.WIDTH,
                        android.provider.MediaStore.Images.ImageColumns.HEIGHT,
                        android.provider.MediaStore.Images.ImageColumns.ORIENTATION,
                        android.provider.MediaStore.Images.ImageColumns.DATE_TAKEN,
                    }, null, null, null)) {
                if (cursor != null && cursor.moveToFirst()) {
                    int iName   = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME);
                    int iSize   = cursor.getColumnIndex(android.provider.OpenableColumns.SIZE);
                    int iMime   = cursor.getColumnIndex(android.provider.MediaStore.Images.ImageColumns.MIME_TYPE);
                    int iW      = cursor.getColumnIndex(android.provider.MediaStore.Images.ImageColumns.WIDTH);
                    int iH      = cursor.getColumnIndex(android.provider.MediaStore.Images.ImageColumns.HEIGHT);
                    int iOrient = cursor.getColumnIndex(android.provider.MediaStore.Images.ImageColumns.ORIENTATION);
                    int iDate   = cursor.getColumnIndex(android.provider.MediaStore.Images.ImageColumns.DATE_TAKEN);

                    String name   = iName   >= 0 ? cursor.getString(iName)  : "n/a";
                    long   size   = iSize   >= 0 ? cursor.getLong(iSize)    : -1;
                    String mime   = iMime   >= 0 ? cursor.getString(iMime)  : "n/a";
                    int    width  = iW      >= 0 ? cursor.getInt(iW)        : -1;
                    int    height = iH      >= 0 ? cursor.getInt(iH)        : -1;
                    int    orient = iOrient >= 0 ? cursor.getInt(iOrient)   : -1;
                    long   date   = iDate   >= 0 ? cursor.getLong(iDate)    : -1;

                    Log.i(TAG, "  File name   : " + name);
                    Log.i(TAG, "  File size   : " + (size >= 0 ? size + " bytes (" + size / 1024 + " KB)" : "n/a"));
                    Log.i(TAG, "  MIME type   : " + mime);
                    Log.i(TAG, "  Stored W×H  : " + (width > 0 && height > 0 ? width + "×" + height : "n/a (not in MediaStore)"));
                    Log.i(TAG, "  EXIF orient : " + (orient >= 0 ? orient + "°" : "n/a"));
                    Log.i(TAG, "  Date taken  : " + (date > 0
                            ? new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss",
                                java.util.Locale.US).format(new java.util.Date(date))
                            : "n/a"));
                } else {
                    Log.i(TAG, "  (no MediaStore metadata — may be a file:// or picker URI)");
                }
            } catch (Exception meta) {
                Log.w(TAG, "  Could not query metadata: " + meta.getMessage());
            }
            Log.i(TAG, "============================");

            Bitmap bmp;
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                Log.d(TAG, "preprocessFromUri: using ImageDecoder (API " + Build.VERSION.SDK_INT + ")");
                ImageDecoder.Source src = ImageDecoder.createSource(
                        context.getContentResolver(), imageUri);
                bmp = ImageDecoder.decodeBitmap(src, (decoder, info, source) ->
                        decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE));
            } else {
                Log.d(TAG, "preprocessFromUri: using MediaStore.getBitmap (API " + Build.VERSION.SDK_INT + ")");
                bmp = MediaStore.Images.Media.getBitmap(context.getContentResolver(), imageUri);
            }
            Log.i(TAG, "Decoded bitmap: " + bmp.getWidth() + "×" + bmp.getHeight()
                    + "  config=" + bmp.getConfig()
                    + "  hasAlpha=" + bmp.hasAlpha()
                    + "  byteCount=" + bmp.getByteCount() + " bytes");

            // Log aspect-ratio mismatch: model expects 504×336 (W×H) = 3:2 landscape.
            // A portrait or square image will be squashed — this affects model quality.
            float srcAspect   = (float) bmp.getWidth() / bmp.getHeight();
            float modelAspect = (float) TARGET_WIDTH    / TARGET_HEIGHT;   // 512/512 = 1.0 (square)
            Log.i(TAG, String.format(
                    "Aspect ratio – source: %.2f (%dx%d)  model target: %.2f (%dx%d)%s",
                    srcAspect, bmp.getWidth(), bmp.getHeight(),
                    modelAspect, TARGET_WIDTH, TARGET_HEIGHT,
                    Math.abs(srcAspect - modelAspect) > 0.3f
                            ? "  *** WARNING: large aspect mismatch – image will be distorted ***"
                            : ""));

            // Ensure ARGB_8888 config required by getPixels()
            if (bmp.getConfig() != Bitmap.Config.ARGB_8888) {
                Log.w(TAG, "Converting bitmap from " + bmp.getConfig() + " to ARGB_8888");
                Bitmap tmp = bmp.copy(Bitmap.Config.ARGB_8888, false);
                bmp.recycle();
                bmp = tmp;
            }
            return preprocessBitmap(context, bmp);
        } catch (Exception e) {
            Log.e(TAG, "Failed to load bitmap from URI", e);
            return null;
        }
    }

    /**
     * Converts a Bitmap to the float32 patchified byte array.
     * Also saves the 512×512 resized input to external storage for debugging.
     *
     * @param context Android context (used for debug image save path)
     * @param src source bitmap (any size)
     * @return float32 byte array [NUM_PATCHES × PATCH_FLOATS × 4 bytes]
     */
    public static byte[] preprocessBitmap(Context context, Bitmap src) {
        Log.d(TAG, "preprocessBitmap: input " + src.getWidth() + "×" + src.getHeight());

        // ── Model input shape summary ─────────────────────────────────────────
        Log.i(TAG, "=== VLM model input spec ===");
        Log.i(TAG, "  Target image size : " + TARGET_WIDTH + "×" + TARGET_HEIGHT + " px (square)");
        Log.i(TAG, "  Patch size        : " + PATCH_SIZE + "×" + PATCH_SIZE + " px");
        Log.i(TAG, "  Spatial grid      : " + PATCH_COLS + " cols × " + PATCH_ROWS + " rows = "
                + NUM_SPATIAL + " patches");
        Log.i(TAG, "  Temporal copies   : " + TEMPORAL_SIZE
                + " (packed per spatial patch as [T,C,H,W])");
        Log.i(TAG, "  Total entries     : " + NUM_PATCHES
                + "  each [" + PATCH_FLOATS + " floats = " + TEMPORAL_SIZE + "×3×"
                + PATCH_SIZE + "×" + PATCH_SIZE + "]");
        Log.i(TAG, "  pixelValues bytes : " + (NUM_PATCHES * PATCH_FLOATS * Float.BYTES)
                + " (" + (NUM_PATCHES * PATCH_FLOATS * Float.BYTES / 1024 / 1024) + " MB)");
        Log.i(TAG, "  pos_cos/sin bytes : " + (NUM_PATCHES * 32 * Float.BYTES)
                + "  [" + NUM_PATCHES + " × 32]");
        Log.i(TAG, "  attn mask bytes   : " + (NUM_SPATIAL * NUM_SPATIAL * Float.BYTES)
                + "  [1 × " + NUM_SPATIAL + " × " + NUM_SPATIAL + "]");
        Log.i(TAG, "============================");

        // 1. Pre-downscale: cap longest edge at 2× model input to avoid OOM on large camera shots
        src = preDownscale(src, Math.max(TARGET_WIDTH, TARGET_HEIGHT) * 2);

        // 2. Center-crop to model aspect ratio (504:336 = 3:2 landscape) before resize.
        //    Without this, portrait or square images are squashed horizontally, causing
        //    the model to see distorted content.
        src = centerCropToAspectRatio(src, TARGET_WIDTH, TARGET_HEIGHT);

        // 3. Resize to exact model input: 504×336
        Bitmap resized = Bitmap.createScaledBitmap(src, TARGET_WIDTH, TARGET_HEIGHT, true);
        Log.d(TAG, "preprocessBitmap: resized to " + resized.getWidth() + "×" + resized.getHeight());

        // ── Save debug image so caller can pull it with adb ──────────────────
        if (context != null) {
            try {
                java.io.File dir  = context.getExternalFilesDir(null);
                java.io.File file = new java.io.File(dir, "vlm_debug_input.png");
                try (java.io.FileOutputStream fos = new java.io.FileOutputStream(file)) {
                    resized.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, fos);
                }
                Log.i(TAG, "Debug input image saved: " + file.getAbsolutePath());
                Log.i(TAG, "  Pull with: adb pull \"" + file.getAbsolutePath() + "\"");
            } catch (Exception e) {
                Log.w(TAG, "Could not save debug image: " + e.getMessage());
            }
        }

        // Read pixels into int[] array (ARGB_8888)
        int[] pixels = new int[TARGET_HEIGHT * TARGET_WIDTH];
        resized.getPixels(pixels, 0, TARGET_WIDTH, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);
        if (resized != src) resized.recycle();

        // Allocate output: [NUM_PATCHES, PATCH_FLOATS] float32
        // Each entry packs both temporal copies: [T=0, C, H, W] || [T=1, C, H, W]
        // = [3×16×16] + [3×16×16] = 768 + 768 = 1536 floats per spatial patch.
        // For a static image the two temporal copies are identical.
        int totalFloats = NUM_PATCHES * PATCH_FLOATS;
        ByteBuffer buf = ByteBuffer.allocate(totalFloats * Float.BYTES);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        FloatBuffer fbuf = buf.asFloatBuffer();

        int pixelsPerChan = PATCH_SIZE * PATCH_SIZE; // 256

        // Build spatial patches in row-major order
        for (int pr = 0; pr < PATCH_ROWS; pr++) {
            for (int pc = 0; pc < PATCH_COLS; pc++) {
                // Build one CHW frame for this patch (768 floats = 3 × 16 × 16).
                // Qwen3-VL vision encoder expects CHW layout per temporal copy:
                //   [R×256, G×256, B×256] — all pixels of one channel first.
                float[] frame = new float[3 * pixelsPerChan];
                int rOff = 0;
                int gOff = pixelsPerChan;
                int bOff = 2 * pixelsPerChan;
                for (int ph = 0; ph < PATCH_SIZE; ph++) {
                    for (int pw = 0; pw < PATCH_SIZE; pw++) {
                        int py = pr * PATCH_SIZE + ph;
                        int px = pc * PATCH_SIZE + pw;
                        int color = pixels[py * TARGET_WIDTH + px];
                        float r = ((color >> 16) & 0xFF) / 255.0f;
                        float g = ((color >>  8) & 0xFF) / 255.0f;
                        float b = ( color        & 0xFF) / 255.0f;
                        int idx = ph * PATCH_SIZE + pw;
                        frame[rOff + idx] = (r - MEAN[0]) / STD[0];
                        frame[gOff + idx] = (g - MEAN[1]) / STD[1];
                        frame[bOff + idx] = (b - MEAN[2]) / STD[2];
                    }
                }
                // Pack as [T=0, C, H, W] || [T=1, C, H, W] — identical for static image
                fbuf.put(frame); // temporal copy 0
                fbuf.put(frame); // temporal copy 1
            }
        }
        // Temporal copies are already packed per spatial patch above.
        // No separate block-level duplication needed (unlike Qwen2.5-VL).

        byte[] result = buf.array();

        // Sanity-check: log min/max/mean of first patch entry (expected range ≈ -1.0 to +1.0).
        // Layout of the first 1536 floats: [R×256, G×256, B×256]_t0 || [R×256, G×256, B×256]_t1
        fbuf.rewind();
        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE, sum = 0;
        int checkCount = PATCH_FLOATS; // first patch only
        for (int i = 0; i < checkCount; i++) {
            float v = fbuf.get();
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        Log.d(TAG, "preprocessBitmap: first patch float stats [T,C,H,W]"
                + " min=" + String.format("%.3f", min)
                + " max=" + String.format("%.3f", max)
                + " mean=" + String.format("%.3f", sum / checkCount));
        Log.d(TAG, "preprocessBitmap: output " + result.length + " bytes"
                + " (expected " + (NUM_PATCHES * PATCH_FLOATS * Float.BYTES) + ")");

        return result;
    }

    /**
     * Proportionally downscales {@code src} so its longest edge does not exceed
     * {@code maxEdge} pixels. Returns the original bitmap unchanged if it already fits.
     * The source bitmap is recycled when a new one is created.
     */
    private static Bitmap preDownscale(Bitmap src, int maxEdge) {
        int maxDim = Math.max(src.getWidth(), src.getHeight());
        if (maxDim <= maxEdge) return src;
        float scale = (float) maxEdge / maxDim;
        int w = Math.max(1, Math.round(src.getWidth()  * scale));
        int h = Math.max(1, Math.round(src.getHeight() * scale));
        Log.d(TAG, "preDownscale: " + src.getWidth() + "×" + src.getHeight()
                + " → " + w + "×" + h + "  (maxEdge=" + maxEdge + ")");
        Bitmap scaled = Bitmap.createScaledBitmap(src, w, h, true);
        src.recycle();
        return scaled;
    }

    /**
     * Center-crops {@code src} to match the {@code targetWidth}:{@code targetHeight}
     * aspect ratio. This prevents content distortion when the source image has a
     * different aspect ratio than the model's required input (504×336, i.e. 3:2 landscape).
     *
     * <p>Examples:
     * <ul>
     *   <li>Portrait 9:16 → crops top/bottom, preserves horizontal center</li>
     *   <li>Square 1:1  → crops top/bottom, preserves horizontal center</li>
     *   <li>Wide 16:9   → crops left/right, preserves vertical center</li>
     * </ul>
     * The source bitmap is recycled when a new one is created.
     */
    private static Bitmap centerCropToAspectRatio(Bitmap src, int targetWidth, int targetHeight) {
        float srcAspect    = (float) src.getWidth()  / src.getHeight();
        float targetAspect = (float) targetWidth / targetHeight;
        if (Math.abs(srcAspect - targetAspect) < 0.02f) return src; // already close enough

        int cropW, cropH;
        if (srcAspect > targetAspect) {
            // Source is wider than target → trim the sides
            cropH = src.getHeight();
            cropW = Math.round(cropH * targetAspect);
        } else {
            // Source is taller than target → trim top and bottom
            cropW = src.getWidth();
            cropH = Math.round(cropW / targetAspect);
        }
        int x = (src.getWidth()  - cropW) / 2;
        int y = (src.getHeight() - cropH) / 2;
        Log.d(TAG, "centerCrop: " + src.getWidth() + "×" + src.getHeight()
                + " → " + cropW + "×" + cropH + " at (" + x + "," + y + ")");
        Bitmap cropped = Bitmap.createBitmap(src, x, y, cropW, cropH);
        src.recycle();
        return cropped;
    }

    /**
     * Loads a raw binary asset file into a byte[].
     * Used for the fixed preprocessing tensors (pos_cos, pos_sin, attn_masks)
     * that are precomputed for the default 336×504 image resolution.
     *
     * @param context  Android context
     * @param assetName asset file name (e.g. "position_ids_cos.raw")
     * @return byte[] contents, or null on error
     */
    public static byte[] loadRawAsset(Context context, String assetName) {
        try (InputStream is = context.getAssets().open(assetName)) {
            byte[] data = new byte[is.available()];
            int read = is.read(data);
            Log.d(TAG, "Loaded asset: " + assetName + " (" + read + " bytes)");
            return data;
        } catch (IOException e) {
            Log.e(TAG, "Failed to load asset: " + assetName, e);
            return null;
        }
    }
}
