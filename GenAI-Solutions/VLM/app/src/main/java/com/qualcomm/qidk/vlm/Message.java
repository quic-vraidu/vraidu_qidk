//============================================================================
// Copyright (c) 2026 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//============================================================================

package com.qualcomm.qidk.vlm;

import android.net.Uri;

/**
 * Represents a single chat message (user or assistant).
 * User messages may carry an optional image URI.
 */
public class Message {
    private final MessageType type;
    private final StringBuilder text;
    private final Uri imageUri;   // null for text-only / assistant messages

    public Message(MessageType type, String text) {
        this(type, text, null);
    }

    public Message(MessageType type, String text, Uri imageUri) {
        this.type     = type;
        this.text     = new StringBuilder(text != null ? text : "");
        this.imageUri = imageUri;
    }

    public MessageType getType()    { return type; }
    public String      getText()    { return text.toString(); }
    public Uri         getImageUri(){ return imageUri; }
    public boolean     hasImage()   { return imageUri != null; }

    /** Appends streaming token fragments to the assistant response. */
    public void append(String token) {
        text.append(token);
    }
}
