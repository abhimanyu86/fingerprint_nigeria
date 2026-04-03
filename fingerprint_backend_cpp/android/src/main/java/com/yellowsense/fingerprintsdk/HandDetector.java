package com.yellowsense.fingerprintsdk;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.annotation.NonNull;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Wraps MediaPipe Hand Landmarker to detect the 4 fingers (INDEX, MIDDLE, RING, LITTLE)
 * in a full camera frame and return cropped JPEG regions ready for the C++ pipeline.
 *
 * Mirrors the reference Python implementation in server.py:
 *   - 4 fingers only (no thumb)
 *   - Full finger bbox: all landmarks from MCP to TIP + 45 px padding
 *   - Extension check: tip.y < mcp.y - 0.04 (normalised)
 *
 * Model file required: hand_landmarker.task in src/main/assets/
 */
public final class HandDetector implements AutoCloseable {

    // ── 4-finger definitions (INDEX, MIDDLE, RING, LITTLE) ──────────────
    // Each entry: { MCP, PIP, DIP, TIP } landmark indices
    private static final int[][] FINGER_LM = {
        {5,  6,  7,  8},   // INDEX
        {9,  10, 11, 12},  // MIDDLE
        {13, 14, 15, 16},  // RING
        {17, 18, 19, 20},  // LITTLE
    };
    private static final String[] FINGER_NAMES = {"Index", "Middle", "Ring", "Little"};

    // ISO/ANSI finger position codes
    // Right: INDEX=2, MIDDLE=3, RING=4, LITTLE=5
    // Left:  INDEX=7, MIDDLE=8, RING=9,  LITTLE=10
    private static final int[] RIGHT_IDS = {2, 3, 4, 5};
    private static final int[] LEFT_IDS  = {7, 8, 9, 10};

    // Pixels of padding added around the full-finger bounding box
    private static final int BBOX_PAD = 45;
    // Minimum crop dimension (px) — smaller crops are noise
    private static final int MIN_CROP = 40;

    private static final String MODEL_ASSET = "hand_landmarker.task";

    private final HandLandmarker landmarker;

    // ── Init ────────────────────────────────────────────────────────────

    public HandDetector(@NonNull Context context) {
        BaseOptions base = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET)
                .build();

        HandLandmarker.HandLandmarkerOptions opts =
                HandLandmarker.HandLandmarkerOptions.builder()
                        .setBaseOptions(base)
                        .setNumHands(1)
                        .setMinHandDetectionConfidence(0.45f)
                        .setMinHandPresenceConfidence(0.45f)
                        .setMinTrackingConfidence(0.45f)
                        .setRunningMode(RunningMode.IMAGE)
                        .build();

        landmarker = HandLandmarker.createFromOptions(context, opts);
    }

    // ── Detection result ─────────────────────────────────────────────────

    public static final class DetectedFinger {
        public final String handedness;      // "Left" or "Right"
        public final String fingerName;      // "Index" | "Middle" | "Ring" | "Little"
        public final int    fingerId;        // ISO/ANSI 1-10
        public final float  detectionScore;  // MediaPipe hand confidence
        public final byte[] croppedJpeg;     // ready for FingerprintSDK.processImage()
        public final int    cropX, cropY, cropW, cropH; // pixel bounds in original image

        DetectedFinger(String handedness, String fingerName, int fingerId,
                       float detectionScore, byte[] croppedJpeg,
                       int cropX, int cropY, int cropW, int cropH) {
            this.handedness     = handedness;
            this.fingerName     = fingerName;
            this.fingerId       = fingerId;
            this.detectionScore = detectionScore;
            this.croppedJpeg    = croppedJpeg;
            this.cropX          = cropX;
            this.cropY          = cropY;
            this.cropW          = cropW;
            this.cropH          = cropH;
        }
    }

    // ── Main API ─────────────────────────────────────────────────────────

    /**
     * Detect extended fingers (INDEX/MIDDLE/RING/LITTLE) in a JPEG frame.
     *
     * @param jpegBytes Full-resolution camera JPEG.
     * @return List of detected fingers (0–4 items). Empty if no hand detected.
     */
    @NonNull
    public List<DetectedFinger> detect(@NonNull byte[] jpegBytes) {
        List<DetectedFinger> results = new ArrayList<>();

        Bitmap bmp = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.length);
        if (bmp == null) return results;

        if (bmp.getConfig() != Bitmap.Config.ARGB_8888) {
            bmp = bmp.copy(Bitmap.Config.ARGB_8888, false);
        }

        MPImage mpImage = new BitmapImageBuilder(bmp).build();
        HandLandmarkerResult detection = landmarker.detect(mpImage);

        int W = bmp.getWidth();
        int H = bmp.getHeight();

        for (int h = 0; h < detection.landmarks().size(); h++) {
            List<NormalizedLandmark> landmarks = detection.landmarks().get(h);

            // Handedness — MediaPipe mirrors front camera labels, flip to match reality
            String handedness = "Right";
            float detScore = 0.88f;
            if (!detection.handednesses().isEmpty() &&
                    h < detection.handednesses().size() &&
                    !detection.handednesses().get(h).isEmpty()) {
                String raw = detection.handednesses().get(h).get(0).categoryName();
                handedness = raw.equals("Left") ? "Right" : "Left";
                detScore   = detection.handednesses().get(h).get(0).score();
            }

            int[] fingerIds = handedness.equals("Right") ? RIGHT_IDS : LEFT_IDS;

            for (int f = 0; f < FINGER_LM.length; f++) {
                int[] ids = FINGER_LM[f];
                NormalizedLandmark mcp = landmarks.get(ids[0]);
                NormalizedLandmark tip = landmarks.get(ids[ids.length - 1]);

                // Extension check: tip.y must be sufficiently above MCP.y
                // (y increases downward in normalised coords)
                if (!(tip.y() < mcp.y() - 0.04f)) continue;

                // Full-finger bounding box: span all landmarks in this finger
                float minX = Float.MAX_VALUE, maxX = Float.MIN_VALUE;
                float minY = Float.MAX_VALUE, maxY = Float.MIN_VALUE;
                for (int id : ids) {
                    NormalizedLandmark lm = landmarks.get(id);
                    minX = Math.min(minX, lm.x());
                    maxX = Math.max(maxX, lm.x());
                    minY = Math.min(minY, lm.y());
                    maxY = Math.max(maxY, lm.y());
                }

                int x1 = Math.max(0, (int)(minX * W) - BBOX_PAD);
                int y1 = Math.max(0, (int)(minY * H) - BBOX_PAD);
                int x2 = Math.min(W, (int)(maxX * W) + BBOX_PAD);
                int y2 = Math.min(H, (int)(maxY * H) + BBOX_PAD);
                int cw = x2 - x1;
                int ch = y2 - y1;

                if (cw < MIN_CROP || ch < MIN_CROP) continue;

                Bitmap crop = Bitmap.createBitmap(bmp, x1, y1, cw, ch);
                byte[] jpeg = bitmapToJpeg(crop, 92);
                crop.recycle();

                results.add(new DetectedFinger(
                        handedness, FINGER_NAMES[f], fingerIds[f],
                        detScore, jpeg,
                        x1, y1, cw, ch));
            }
        }

        bmp.recycle();
        return results;
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static byte[] bitmapToJpeg(Bitmap bmp, int quality) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, quality, out);
        return out.toByteArray();
    }

    @Override
    public void close() {
        landmarker.close();
    }
}
