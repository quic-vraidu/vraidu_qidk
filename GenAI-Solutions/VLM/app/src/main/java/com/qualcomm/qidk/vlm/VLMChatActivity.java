//============================================================================
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//============================================================================

package com.qualcomm.qidk.vlm;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.recyclerview.widget.DefaultItemAnimator;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

/**
 * VLM Chat screen.
 *
 * <p>Supports two inference paths:
 * <ol>
 *   <li><b>Text-only</b>: {@link #runTextInference(String, TokenCallback)} –
 *       sends the full prompt to the lutEncoder node only.</li>
 *   <li><b>Image + Text (VLM)</b>: {@link #runVLMInference} –
 *       preprocesses the image via {@link ImagePreprocessor}, loads the fixed
 *       RoPE/mask tensors from assets, and feeds all inputs to the 3-node
 *       GeniePipeline (imageEncoder + lutEncoder + textGenerator).</li>
 * </ol>
 */
public class VLMChatActivity extends AppCompatActivity {

    private static final String TAG             = "VLM_ChatActivity";
    private static final int    REQ_PERMISSION  = 101;

    // ── JNI (native-lib.cpp) ─────────────────────────────────────────────────

    /** Streaming token callback interface invoked from native code. */
    public interface TokenCallback {
        void onToken(String token);
    }

    /**
     * Full VLM inference: image + text.
     * @param prefix      prompt prefix ending in {@code <|vision_start|>}
     * @param pixelValues float32 patchified image [1728×588] as byte[]
     * @param posCos      float32 RoPE cos positions [1728×20] as byte[]
     * @param posSin      float32 RoPE sin positions [1728×20] as byte[]
     * @param windowMask  float32 window attention mask [864×864] as byte[]
     * @param fullMask    float32 full   attention mask [864×864] as byte[]
     * @param suffix      prompt suffix starting with {@code <|vision_end|>...}
     * @param callback    streaming token callback
     */
    public native int runVLMInference(String prefix,
                                      byte[] pixelValues,
                                      byte[] posCos,
                                      byte[] posSin,
                                      byte[] windowMask,
                                      byte[] fullMask,
                                      String suffix,
                                      TokenCallback callback);

    /** Text-only inference: single prompt string routed via lutEncoder. */
    public native int runTextInference(String prompt, TokenCallback callback);

    /** Resets the KV-cache / conversation context. */
    public native int resetPipeline();

    // ── UI ──────────────────────────────────────────────────────────────────
    private RecyclerView       recyclerView;
    private MessageListAdapter adapter;
    private LinearLayoutManager layoutManager;
    private EditText      etInput;
    private ImageButton   btnSend, btnAttachImage, btnClearCache, btnRemoveImage;
    private LinearLayout  layoutImagePreview, layoutInputBar;
    private ImageView     ivPreview;
    private TextView      tvImageName;

    // ── State ────────────────────────────────────────────────────────────────
    private Uri     selectedImageUri  = null;
    private boolean isInferenceRunning = false;

    // ── Fixed preprocessing tensors (loaded once from assets) ────────────────
    private byte[] g_posCos;
    private byte[] g_posSin;
    private byte[] g_windowMask;
    private byte[] g_fullMask;

    // ── Image picker ─────────────────────────────────────────────────────────
    private final ActivityResultLauncher<Intent> imagePickerLauncher =
            registerForActivityResult(
                    new ActivityResultContracts.StartActivityForResult(),
                    result -> {
                        if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                            Uri uri = result.getData().getData();
                            if (uri != null) onImageSelected(uri);
                        }
                    });

    // ── Lifecycle ────────────────────────────────────────────────────────────

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_vlm_chat);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            Insets ime  = insets.getInsets(WindowInsetsCompat.Type.ime());
            v.setPadding(bars.left, bars.top, bars.right, Math.max(bars.bottom, ime.bottom));
            return insets;
        });

        initViews();
        initRecyclerView();
        setListeners();

        // Load fixed preprocessing tensors from assets (precomputed for 336×504)
        loadFixedAssets();
    }

    // ── Initialisation ───────────────────────────────────────────────────────

    private void initViews() {
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        toolbar.setNavigationOnClickListener(v -> finish());

        recyclerView       = findViewById(R.id.recyclerViewChat);
        etInput            = findViewById(R.id.etInput);
        btnSend            = findViewById(R.id.btnSend);
        btnAttachImage     = findViewById(R.id.btnAttachImage);
        btnClearCache      = findViewById(R.id.btnClearCache);
        layoutImagePreview = findViewById(R.id.layoutImagePreview);
        ivPreview          = findViewById(R.id.ivPreview);
        tvImageName        = findViewById(R.id.tvImageName);
        btnRemoveImage     = findViewById(R.id.btnRemoveImage);
        layoutInputBar     = findViewById(R.id.layoutInputBar);
    }

    private void initRecyclerView() {
        adapter       = new MessageListAdapter();
        adapter.setContext(this);
        layoutManager = new LinearLayoutManager(this);
        layoutManager.setStackFromEnd(true);
        recyclerView.setLayoutManager(layoutManager);
        recyclerView.setAdapter(adapter);
        // Disable item-change crossfade animation to prevent flickering during token streaming
        DefaultItemAnimator animator = (DefaultItemAnimator) recyclerView.getItemAnimator();
        if (animator != null) animator.setSupportsChangeAnimations(false);
    }

    private void setListeners() {
        btnSend.setOnClickListener(v -> { if (!isInferenceRunning) handleSend(); });
        btnAttachImage.setOnClickListener(v -> requestImagePick());
        btnRemoveImage.setOnClickListener(v -> clearSelectedImage());
        btnClearCache.setOnClickListener(v -> {
            adapter.clearMessages();
            int status = resetPipeline();
            Toast.makeText(this,
                    status == 0 ? getString(R.string.chat_cleared) : "Reset failed (" + status + ")",
                    Toast.LENGTH_SHORT).show();
        });
    }

    private void loadFixedAssets() {
        // These tensors are precomputed for the 336×504 default image size.
        // They correspond to the Qwen2.5-VL MRoPE positions and attention patterns.
        new Thread(() -> {
            g_posCos     = ImagePreprocessor.loadRawAsset(this, "position_ids_cos.raw");
            g_posSin     = ImagePreprocessor.loadRawAsset(this, "position_ids_sin.raw");
            g_windowMask = ImagePreprocessor.loadRawAsset(this, "window_attention_mask.raw");
            g_fullMask   = ImagePreprocessor.loadRawAsset(this, "full_attention_mask.raw");
            Log.d(TAG, "Fixed assets loaded: cos=" + (g_posCos != null ? g_posCos.length : "null")
                     + " sin=" + (g_posSin != null ? g_posSin.length : "null")
                     + " window=" + (g_windowMask != null ? g_windowMask.length : "null")
                     + " full=" + (g_fullMask != null ? g_fullMask.length : "null"));
        }).start();
    }

    // ── Send / Inference ─────────────────────────────────────────────────────

    private void handleSend() {
        String userText  = etInput.getText().toString().trim();
        boolean hasImage = selectedImageUri != null;

        if (TextUtils.isEmpty(userText) && !hasImage) {
            Toast.makeText(this, "Enter a message or attach an image.", Toast.LENGTH_SHORT).show();
            return;
        }
        if (TextUtils.isEmpty(userText)) userText = "Describe this image.";

        // Add user message bubble
        Message userMsg = new Message(MessageType.USER, userText, selectedImageUri);
        adapter.addMessage(userMsg);
        scrollToBottom();

        final String finalText = userText;
        final Uri    imageUri  = selectedImageUri;

        etInput.setText("");
        clearSelectedImage();

        // Add empty assistant bubble (will be filled by streaming)
        adapter.addMessage(new Message(MessageType.ASSISTANT, ""));
        scrollToBottom();

        setInputEnabled(false);
        isInferenceRunning = true;

        if (hasImage) {
            runVLMAsync(imageUri, finalText);
        } else {
            runTextAsync(finalText);
        }
    }

    // ── VLM image + text inference ───────────────────────────────────────────

    private void runVLMAsync(Uri imageUri, String userText) {
        new Thread(() -> {
            try {
                // 0. Reset KV-cache before every VLM call.
                //    The image position encodings (pos_cos / pos_sin) are fixed assets
                //    that always start at position 0.  Without a reset the second image's
                //    patch embeddings land at positions 0-1727 inside an already-populated
                //    KV-cache, causing the model to see them as corrupted/distorted.
                int resetStatus = resetPipeline();
                Log.i(TAG, "Auto-reset before VLM inference: status=" + resetStatus);

                // 1. Preprocess image → float32 patches [1728×588]
                byte[] pixelValues = ImagePreprocessor.preprocessFromUri(this, imageUri);
                if (pixelValues == null) {
                    appendError("Image preprocessing failed.");
                    return;
                }
                Log.d(TAG, "pixelValues: " + pixelValues.length + " bytes");

                // 2. Ensure fixed tensors are loaded
                if (g_posCos == null || g_posSin == null || g_windowMask == null || g_fullMask == null) {
                    appendError("Fixed preprocessing tensors not loaded yet. Retry.");
                    return;
                }
                Log.d(TAG, "Fixed tensors: posCos=" + g_posCos.length
                        + " posSin=" + g_posSin.length
                        + " windowMask=" + g_windowMask.length
                        + " fullMask=" + g_fullMask.length + " bytes");

                // 3. Build prefix / suffix using Qwen2.5-VL template
                String prefix = VLMHelper.buildVLMPrefix();
                String suffix = VLMHelper.buildVLMSuffix(userText);
                Log.d(TAG, "VLM prefix (" + prefix.length() + " chars): " + prefix);
                Log.d(TAG, "VLM suffix (" + suffix.length() + " chars): " + suffix);

                // 4. Run VLM pipeline inference
                final int[]          tokenCount  = {0};
                final StringBuilder  fullResponse = new StringBuilder();
                TokenCallback cb = token -> {
                    tokenCount[0]++;
                    fullResponse.append(token);
                    if (tokenCount[0] <= 5 || tokenCount[0] % 50 == 0) {
                        Log.d(TAG, "token #" + tokenCount[0] + ": \"" + token + "\"");
                    }
                    runOnUiThread(() -> {
                        adapter.appendToLastMessage(token);
                        scrollToBottomInstant();
                    });
                };

                Log.i(TAG, "Starting VLM inference...");
                long t0 = System.currentTimeMillis();
                int status = runVLMInference(prefix, pixelValues,
                                             g_posCos, g_posSin,
                                             g_windowMask, g_fullMask,
                                             suffix, cb);
                long elapsed = System.currentTimeMillis() - t0;
                Log.i(TAG, "VLM inference done: status=" + status
                        + " tokens=" + tokenCount[0]
                        + " elapsed=" + elapsed + "ms");
                Log.d(TAG, "Full VLM response: \"" + fullResponse + "\"");

                if (status != 0) appendError("VLM inference error: " + status);

            } catch (Exception e) {
                Log.e(TAG, "VLM inference exception", e);
                appendError("Exception: " + e.getMessage());
            } finally {
                runOnUiThread(() -> {
                    isInferenceRunning = false;
                    setInputEnabled(true);
                });
            }
        }).start();
    }

    // ── Text-only inference ──────────────────────────────────────────────────

    private void runTextAsync(String userText) {
        new Thread(() -> {
            try {
                String prompt = VLMHelper.buildTextPrompt(userText);
                Log.d(TAG, "Text prompt (" + prompt.length() + " chars): " + prompt);

                final int[]         tokenCount   = {0};
                final StringBuilder fullResponse = new StringBuilder();
                TokenCallback cb = token -> {
                    tokenCount[0]++;
                    fullResponse.append(token);
                    if (tokenCount[0] <= 5 || tokenCount[0] % 50 == 0) {
                        Log.d(TAG, "token #" + tokenCount[0] + ": \"" + token + "\"");
                    }
                    runOnUiThread(() -> {
                        adapter.appendToLastMessage(token);
                        scrollToBottomInstant();
                    });
                };

                Log.i(TAG, "Starting text-only inference...");
                long t0 = System.currentTimeMillis();
                int status = runTextInference(prompt, cb);
                long elapsed = System.currentTimeMillis() - t0;
                Log.i(TAG, "Text inference done: status=" + status
                        + " tokens=" + tokenCount[0]
                        + " elapsed=" + elapsed + "ms");
                Log.d(TAG, "Full text response: \"" + fullResponse + "\"");

                if (status != 0) appendError("Text inference error: " + status);
            } catch (Exception e) {
                Log.e(TAG, "Text inference exception", e);
                appendError("Exception: " + e.getMessage());
            } finally {
                runOnUiThread(() -> {
                    isInferenceRunning = false;
                    setInputEnabled(true);
                });
            }
        }).start();
    }

    // ── Image picker ─────────────────────────────────────────────────────────

    private void requestImagePick() {
        String perm = (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
                ? Manifest.permission.READ_MEDIA_IMAGES
                : Manifest.permission.READ_EXTERNAL_STORAGE;
        if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{perm}, REQ_PERMISSION);
            return;
        }
        openGallery();
    }

    @Override
    public void onRequestPermissionsResult(int req, String[] permissions, int[] grants) {
        super.onRequestPermissionsResult(req, permissions, grants);
        if (req == REQ_PERMISSION && grants.length > 0 && grants[0] == PackageManager.PERMISSION_GRANTED)
            openGallery();
        else
            Toast.makeText(this, "Gallery permission denied.", Toast.LENGTH_SHORT).show();
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        imagePickerLauncher.launch(intent);
    }

    private void onImageSelected(Uri uri) {
        selectedImageUri = uri;
        layoutImagePreview.setVisibility(View.VISIBLE);
        String name = VLMHelper.getFileName(uri);
        tvImageName.setText(name != null ? name : uri.toString());

        // Load a 1/4-resolution thumbnail on a background thread to avoid
        // holding the full-resolution bitmap in memory and blocking the UI thread.
        new Thread(() -> {
            try {
                ImageDecoder.Source src = ImageDecoder.createSource(getContentResolver(), uri);
                Bitmap thumb = ImageDecoder.decodeBitmap(src,
                        (decoder, info, source) -> decoder.setTargetSampleSize(4));
                runOnUiThread(() -> ivPreview.setImageBitmap(thumb));
            } catch (Exception e) {
                Log.w(TAG, "Thumbnail decode failed, falling back to setImageURI: " + e.getMessage());
                runOnUiThread(() -> ivPreview.setImageURI(uri));
            }
        }).start();
    }

    private void clearSelectedImage() {
        selectedImageUri = null;
        layoutImagePreview.setVisibility(View.GONE);
        ivPreview.setImageDrawable(null);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private void setInputEnabled(boolean enabled) {
        etInput.setEnabled(enabled);
        btnSend.setEnabled(enabled);
        btnAttachImage.setEnabled(enabled);
    }

    private void scrollToBottom() {
        int n = adapter.getItemCount();
        if (n > 0) recyclerView.smoothScrollToPosition(n - 1);
    }

    /** Instant (no animation) scroll — used during token streaming to avoid competing animations. */
    private void scrollToBottomInstant() {
        int n = adapter.getItemCount();
        if (n > 0) recyclerView.scrollToPosition(n - 1);
    }

    private void appendError(String msg) {
        runOnUiThread(() -> adapter.appendToLastMessage("\n[Error: " + msg + "]"));
        Log.e(TAG, msg);
    }
}
