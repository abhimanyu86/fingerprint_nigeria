package com.yellowsense.fingerprintsdk;

import android.content.Context;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Main entry-point for the Fingerprint SDK.
 *
 * ── Quick-start ───────────────────────────────────────────────────────
 *
 * 1. Initialize once (e.g. in Application.onCreate):
 *      FingerprintSDK.init(context);
 *
 * 2. Process a full camera frame (hands auto-detected):
 *      List<FingerResult> results = FingerprintSDK.processFrame(jpegBytes);
 *      for (FingerResult r : results) {
 *          Log.d("SDK", r.toString());
 *          if (r.accepted) saveTemplate(r.template);
 *      }
 *
 * 3. Or process a pre-cropped finger image directly:
 *      Map<String,Object> r = FingerprintSDK.processImage(jpegBytes);
 *
 * 4. Match two templates:
 *      float score = FingerprintSDK.matchTemplates(templateA, templateB);
 *
 * ── Model asset required ──────────────────────────────────────────────
 * Place hand_landmarker.task in src/main/assets/ of your app module.
 * Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/
 *           hand_landmarker/float16/latest/hand_landmarker.task
 */
public final class FingerprintSDK {

    static {
        System.loadLibrary("fingerprint_sdk");
    }

    private FingerprintSDK() {}

    // ── Singleton HandDetector (lazy init) ────────────────────────────
    @Nullable private static volatile HandDetector handDetector;
    private static final Object INIT_LOCK = new Object();

    /**
     * Initialize the SDK with application context.
     * Must be called before processFrame(). Safe to call multiple times.
     *
     * @param context Application or Activity context
     */
    public static void init(@NonNull Context context) {
        if (handDetector == null) {
            synchronized (INIT_LOCK) {
                if (handDetector == null) {
                    handDetector = new HandDetector(context.getApplicationContext());
                }
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════
    // NEW: Full-frame pipeline (hand detection + crop + fingerprint)
    // ═════════════════════════════════════════════════════════════════

    /**
     * Process a full camera JPEG frame.
     *
     * Internally:
     *   1. MediaPipe detects up to 2 hands and all 10 finger landmarks.
     *   2. Each fingertip region is cropped automatically.
     *   3. The fingerprint pipeline (quality → liveness → template) runs on each crop.
     *
     * @param jpegBytes Full-resolution JPEG from camera (any orientation)
     * @return List of FingerResult, one per detected finger (0-10 items).
     *         Empty if no hand detected or SDK not initialized.
     * @throws IllegalStateException if init() was not called first
     */
    @NonNull
    public static List<FingerResult> processFrame(@NonNull byte[] jpegBytes) {
        HandDetector detector = handDetector;
        if (detector == null) {
            throw new IllegalStateException(
                    "FingerprintSDK.init(context) must be called before processFrame()");
        }

        List<HandDetector.DetectedFinger> fingers = detector.detect(jpegBytes);
        List<FingerResult> results = new ArrayList<>(fingers.size());

        for (HandDetector.DetectedFinger finger : fingers) {
            Map<String, Object> pipeline = processImageWithConfidence(
                    finger.croppedJpeg, finger.detectionScore);
            results.add(new FingerResult(
                    finger.handedness,
                    finger.fingerName,
                    finger.fingerId,
                    finger.detectionScore,
                    finger.cropX, finger.cropY, finger.cropW, finger.cropH,
                    pipeline));
        }

        return results;
    }

    /**
     * Convenience: returns only fingers that passed the full pipeline
     * (quality ACCEPT + liveness pass + template extracted).
     */
    @NonNull
    public static List<FingerResult> processFrameAcceptedOnly(@NonNull byte[] jpegBytes) {
        List<FingerResult> all = processFrame(jpegBytes);
        List<FingerResult> ok  = new ArrayList<>();
        for (FingerResult r : all) {
            if (r.accepted) ok.add(r);
        }
        return ok;
    }

    // ═════════════════════════════════════════════════════════════════
    // EXISTING: Pre-cropped finger image pipeline
    // ═════════════════════════════════════════════════════════════════

    /**
     * Run the full pipeline on a pre-cropped JPEG finger image.
     *
     * Returns a Map with keys:
     *   qualityScore, blurScore, contrastScore, ridgeClarityScore,
     *   coverageScore, orientationScore, guidance,
     *   livenessScore, isLive, glareDetected, textureScore, skinScore,
     *   accepted, template, templateSuccess, errorMessage
     */
    @NonNull
    public static Map<String, Object> processImage(byte[] imageBytes) {
        return processImageWithConfidence(imageBytes, 0.88f);
    }

    /**
     * Same as processImage() but also passes the MediaPipe hand detection
     * confidence score so the liveness detector can use it as its base.
     */
    @NonNull
    public static native Map<String, Object> processImageWithConfidence(
            byte[] imageBytes, float handConfidence);

    /**
     * Match two ISO 19794-2 base64 templates.
     *
     * @return score 0.0–1.0  (threshold for match: 0.35)
     */
    public static native float matchTemplates(@NonNull String templateA,
                                              @NonNull String templateB);

    /**
     * Quick quality score only (skips liveness + template encoding).
     *
     * @return composite quality score 0-100
     */
    public static native float getQualityScore(byte[] imageBytes);

    /**
     * Convenience: returns true when match score >= 0.35.
     */
    public static boolean isMatch(@NonNull String templateA,
                                  @NonNull String templateB) {
        return matchTemplates(templateA, templateB) >= 0.35f;
    }

    // ── Cleanup ───────────────────────────────────────────────────────

    /**
     * Release MediaPipe resources. Call from Application.onTerminate()
     * or when the SDK is no longer needed.
     */
    public static void release() {
        synchronized (INIT_LOCK) {
            if (handDetector != null) {
                handDetector.close();
                handDetector = null;
            }
        }
    }
}
