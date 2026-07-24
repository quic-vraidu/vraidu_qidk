//============================================================================
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//============================================================================

package com.qualcomm.qidk.vlm;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;

/**
 * Entry-point activity.  Loads the full VLM pipeline (3 nodes: imageEncoder,
 * lutEncoder, textGenerator) then launches VLMChatActivity.
 *
 * <p>Device setup required before first launch:
 * <pre>
 *   adb root && adb remount
 *   adb shell mkdir -p /data/local/tmp/genie_bundle
 *   adb push &lt;qwen_bundle&gt;/* /data/local/tmp/genie_bundle/
 *   adb push libGenie.so libQnnHtp.so /data/local/tmp/genie_bundle/
 *   adb shell setenforce 0
 * </pre>
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "VLM_MainActivity";

    static { System.loadLibrary("vlmassistant"); }

    // JNI declarations (native-lib.cpp)
    public native int loadPipeline(String imgEncJson, String lutEncJson,
                                   String textGenJson, String nativeLibPath);
    public native int freePipeline();

    // ── UI ──────────────────────────────────────────────────────────────────
    private ProgressBar  progressBar;
    private TextView     tvStatus;
    private ExtendedFloatingActionButton fab;
    private com.google.android.material.button.MaterialButton btnLoad;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets bars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return insets;
        });
        setSupportActionBar((Toolbar) findViewById(R.id.toolbar));

        progressBar = findViewById(R.id.progressBar);
        tvStatus    = findViewById(R.id.tvStatus);
        fab         = findViewById(R.id.fab);
        btnLoad     = findViewById(R.id.btnLoadModel);

        fab.setEnabled(false);
        fab.setAlpha(0.5f);

        btnLoad.setOnClickListener(v -> loadPipelineAsync());
        fab.setOnClickListener(v -> startActivity(
                new Intent(this, VLMChatActivity.class)));

        // Auto-load on startup
        loadPipelineAsync();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        freePipeline();
    }

    // ── Pipeline loading (background thread) ────────────────────────────────
    private void loadPipelineAsync() {
        btnLoad.setEnabled(false);
        progressBar.setVisibility(View.VISIBLE);
        setStatus(getString(R.string.model_loading), getColor(R.color.status_loading));

        new Thread(() -> {
            try {
                // Read 3 node config JSONs from assets
                String imgEncJson  = VLMHelper.loadTextAsset(this, "img-enc-htp.json");
                String lutEncJson  = VLMHelper.loadTextAsset(this, "text-encoder.json");
                String textGenJson = VLMHelper.loadTextAsset(this, "text-generator.json");

                if (imgEncJson == null || lutEncJson == null || textGenJson == null) {
                    setStatusUi(getString(R.string.model_load_failed), R.color.status_error);
                    return;
                }

                String nativeLibPath = getApplicationInfo().nativeLibraryDir;
                Log.d(TAG, "nativeLibPath: " + nativeLibPath);

                int status = loadPipeline(imgEncJson, lutEncJson, textGenJson, nativeLibPath);
                Log.d(TAG, "loadPipeline status: " + status);

                if (status == 0) {
                    setStatusUi(getString(R.string.model_loaded), R.color.status_ok);
                    runOnUiThread(() -> {
                        fab.setEnabled(true);
                        fab.setAlpha(1.0f);
                    });
                } else {
                    setStatusUi(getString(R.string.model_load_failed) + " (code: " + status + ")",
                                R.color.status_error);
                }
            } catch (Exception e) {
                Log.e(TAG, "Exception during pipeline load", e);
                setStatusUi(getString(R.string.model_load_failed), R.color.status_error);
            } finally {
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    btnLoad.setEnabled(true);
                });
            }
        }).start();
    }

    private void setStatus(String text, int color) {
        if (tvStatus != null) { tvStatus.setText(text); tvStatus.setTextColor(color); }
    }
    private void setStatusUi(String text, int colorRes) {
        runOnUiThread(() -> setStatus(text, getColor(colorRes)));
    }
}
