package com.yellowsense.fingerprintsdk;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * Result of running the full fingerprint pipeline on one detected finger.
 * Produced by FingerprintSDK.processFrame().
 */
public final class FingerResult {

    // ── Finger identity ───────────────────────────────────────────────
    /** "Left" or "Right" */
    public final String handedness;

    /** "Thumb" | "Index" | "Middle" | "Ring" | "Pinky" */
    public final String fingerName;

    /**
     * ISO/ANSI finger position code.
     * Right: Thumb=1, Index=2, Middle=3, Ring=4, Pinky=5
     * Left:  Thumb=6, Index=7, Middle=8, Ring=9, Pinky=10
     */
    public final int fingerId;

    /** MediaPipe hand detection confidence for this hand (0.0–1.0) */
    public final float detectionConfidence;

    // ── Crop geometry (pixels in the original frame) ──────────────────
    public final int cropX, cropY, cropWidth, cropHeight;

    // ── Quality ───────────────────────────────────────────────────────
    public final float qualityScore;
    public final float blurScore;
    public final float contrastScore;
    public final float ridgeClarityScore;
    public final float coverageScore;
    public final float orientationScore;
    public final String guidance;
    /** "ACCEPT" | "RETRY" | "REJECT" */
    public final String qualityDecision;

    // ── Liveness ──────────────────────────────────────────────────────
    public final float livenessScore;
    public final boolean isLive;
    public final boolean glareDetected;
    public final float textureScore;
    public final float skinScore;

    // ── Template ──────────────────────────────────────────────────────
    public final boolean templateSuccess;
    @Nullable public final String template;  // base64 ISO 19794-2, null if failed

    // ── Overall ───────────────────────────────────────────────────────
    public final boolean accepted;
    @NonNull public final String errorMessage; // empty string on success

    // ── Constructor (built by HandDetector) ───────────────────────────
    FingerResult(
            String handedness, String fingerName, int fingerId,
            float detectionConfidence,
            int cropX, int cropY, int cropWidth, int cropHeight,
            java.util.Map<String, Object> pipelineResult) {
        this.handedness          = handedness;
        this.fingerName          = fingerName;
        this.fingerId            = fingerId;
        this.detectionConfidence = detectionConfidence;
        this.cropX               = cropX;
        this.cropY               = cropY;
        this.cropWidth           = cropWidth;
        this.cropHeight          = cropHeight;

        this.qualityScore        = getFloat(pipelineResult, "qualityScore");
        this.blurScore           = getFloat(pipelineResult, "blurScore");
        this.contrastScore       = getFloat(pipelineResult, "contrastScore");
        this.ridgeClarityScore   = getFloat(pipelineResult, "ridgeClarityScore");
        this.coverageScore       = getFloat(pipelineResult, "coverageScore");
        this.orientationScore    = getFloat(pipelineResult, "orientationScore");
        this.guidance            = getString(pipelineResult, "guidance");
        this.qualityDecision     = getString(pipelineResult, "guidance"); // updated below
        this.livenessScore       = getFloat(pipelineResult, "livenessScore");
        this.isLive              = getBool(pipelineResult, "isLive");
        this.glareDetected       = getBool(pipelineResult, "glareDetected");
        this.textureScore        = getFloat(pipelineResult, "textureScore");
        this.skinScore           = getFloat(pipelineResult, "skinScore");
        this.templateSuccess     = getBool(pipelineResult, "templateSuccess");
        String tmpl              = getString(pipelineResult, "template");
        this.template            = (tmpl != null && !tmpl.isEmpty()) ? tmpl : null;
        this.accepted            = getBool(pipelineResult, "accepted");
        this.errorMessage        = getString(pipelineResult, "errorMessage");
    }

    private static float getFloat(java.util.Map<String,Object> m, String key) {
        Object v = m.get(key);
        return (v instanceof Float) ? (Float) v : 0f;
    }
    private static boolean getBool(java.util.Map<String,Object> m, String key) {
        Object v = m.get(key);
        return (v instanceof Boolean) && (Boolean) v;
    }
    private static String getString(java.util.Map<String,Object> m, String key) {
        Object v = m.get(key);
        return (v instanceof String) ? (String) v : "";
    }

    @NonNull
    @Override
    public String toString() {
        return String.format("[%s %s (id=%d)] quality=%.1f liveness=%.2f accepted=%b",
                handedness, fingerName, fingerId, qualityScore, livenessScore, accepted);
    }
}
