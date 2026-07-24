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
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.cardview.widget.CardView;
import androidx.recyclerview.widget.RecyclerView;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * RecyclerView adapter for the VLM chat message list.
 * Supports two view types: USER (right-aligned, optional image) and ASSISTANT (left-aligned, streaming).
 */
public class MessageListAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {

    private static final int VIEW_TYPE_USER      = 0;
    private static final int VIEW_TYPE_ASSISTANT = 1;

    private final List<Message> messages = new ArrayList<>();
    private Context context;

    public void setContext(Context ctx) { this.context = ctx; }

    public void addMessage(Message msg) {
        messages.add(msg);
        notifyItemInserted(messages.size() - 1);
    }

    /** Clears all messages from the chat list. */
    public void clearMessages() {
        int count = messages.size();
        messages.clear();
        notifyItemRangeRemoved(0, count);
    }

    /**
     * Appends a token fragment to the last message (assistant streaming).
     * Shows the progress bar until the first token arrives.
     */
    public void appendToLastMessage(String token) {
        if (messages.isEmpty()) return;
        int idx = messages.size() - 1;
        messages.get(idx).append(token);
        notifyItemChanged(idx);
    }

    public int getMessageCount() { return messages.size(); }

    @Override
    public int getItemCount() { return messages.size(); }

    @Override
    public int getItemViewType(int position) {
        return messages.get(position).getType() == MessageType.USER
                ? VIEW_TYPE_USER
                : VIEW_TYPE_ASSISTANT;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        LayoutInflater inflater = LayoutInflater.from(parent.getContext());
        if (viewType == VIEW_TYPE_USER) {
            View v = inflater.inflate(R.layout.item_message_user, parent, false);
            return new UserViewHolder(v);
        } else {
            View v = inflater.inflate(R.layout.item_message_assistant, parent, false);
            return new AssistantViewHolder(v);
        }
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        Message msg = messages.get(position);
        if (holder instanceof UserViewHolder) {
            ((UserViewHolder) holder).bind(msg, context);
        } else {
            ((AssistantViewHolder) holder).bind(msg);
        }
    }

    // -------------------------------------------------------------------------
    // ViewHolder: User
    // -------------------------------------------------------------------------
    static class UserViewHolder extends RecyclerView.ViewHolder {
        final TextView  tvText;
        final ImageView ivImage;
        final CardView  cardImage;

        UserViewHolder(View itemView) {
            super(itemView);
            tvText    = itemView.findViewById(R.id.tvUserText);
            ivImage   = itemView.findViewById(R.id.ivUserImage);
            cardImage = itemView.findViewById(R.id.cardUserImage);
        }

        void bind(Message msg, Context ctx) {
            tvText.setText(msg.getText());
            if (msg.hasImage() && ctx != null) {
                cardImage.setVisibility(View.VISIBLE);
                try {
                    Bitmap bmp;
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                        // ImageDecoder handles EXIF orientation automatically
                        ImageDecoder.Source src = ImageDecoder.createSource(
                                ctx.getContentResolver(), msg.getImageUri());
                        bmp = ImageDecoder.decodeBitmap(src,
                                (decoder, info, source) -> decoder.setTargetSampleSize(4));
                    } else {
                        BitmapFactory.Options opts = new BitmapFactory.Options();
                        opts.inSampleSize = 4;
                        try (InputStream is = ctx.getContentResolver()
                                .openInputStream(msg.getImageUri())) {
                            bmp = BitmapFactory.decodeStream(is, null, opts);
                        }
                    }
                    ivImage.setImageBitmap(bmp);
                } catch (Exception e) {
                    ivImage.setImageURI(msg.getImageUri());
                }
            } else {
                cardImage.setVisibility(View.GONE);
            }
        }
    }

    // -------------------------------------------------------------------------
    // ViewHolder: Assistant
    // -------------------------------------------------------------------------
    static class AssistantViewHolder extends RecyclerView.ViewHolder {
        final TextView   tvText;
        final ProgressBar progressThinking;

        AssistantViewHolder(View itemView) {
            super(itemView);
            tvText           = itemView.findViewById(R.id.tvAssistantText);
            progressThinking = itemView.findViewById(R.id.progressThinking);
        }

        void bind(Message msg) {
            String text = msg.getText();
            if (text.isEmpty()) {
                // Still streaming – show spinner
                progressThinking.setVisibility(View.VISIBLE);
                tvText.setText("");
            } else {
                progressThinking.setVisibility(View.GONE);
                tvText.setText(text);
            }
        }
    }
}
