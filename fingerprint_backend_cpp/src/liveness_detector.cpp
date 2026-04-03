#include "liveness_detector.h"
#include <cmath>
#include <algorithm>

namespace fingerprint {

float LivenessDetector::checkGlare(const cv::Mat& gray) {
    // Bright pixels > 245 must stay under 12 % of frame
    cv::Mat bright;
    cv::threshold(gray, bright, 245, 255, cv::THRESH_BINARY);
    double ratio = (double)cv::countNonZero(bright) / (double)bright.total();
    // Returns 1.0 when no glare; goes negative when ratio >= 12 %
    return (float)(1.0 - ratio / 0.12);
}

float LivenessDetector::computeTextureScore(const cv::Mat& gray) {
    // Local texture variance via box filter: Var = E[x²] - E[x]²
    cv::Mat gF;
    gray.convertTo(gF, CV_32F);

    cv::Mat gF2 = gF.mul(gF);
    cv::Mat Ex, Ex2;
    cv::boxFilter(gF,  Ex,  CV_32F, cv::Size(5, 5));
    cv::boxFilter(gF2, Ex2, CV_32F, cv::Size(5, 5));

    cv::Mat variance = Ex2 - Ex.mul(Ex);
    cv::Scalar meanVar = cv::mean(variance);
    double localStd = std::sqrt(std::abs(meanVar[0]));

    // Real skin on phone camera: localStd 15-90 → 1.0
    // High-res cameras produce sharper images so cap raised to 120
    // Printed / screen: localStd 3-12 → 0.3
    if (localStd < 3  || localStd > 120) return 0.3f;
    if (localStd >= 15 && localStd <= 90) return 1.0f;
    if (localStd < 15) return (float)(localStd / 15.0);
    return std::max(0.3f, (float)(1.0 - (localStd - 90.0) / 120.0));
}

float LivenessDetector::computeSkinScore(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    // OpenCV HSV: H 0-180, S 0-255, V 0-255
    // Extended skin hue: 0-30 and 160-180 (wraps), S 15-220, V 40-255
    // Covers light, olive, and darker South-Asian / African skin tones
    cv::Mat mask1, mask2, skinMask;
    cv::inRange(hsv, cv::Scalar(0,   15,  40), cv::Scalar(30,  220, 255), mask1);
    cv::inRange(hsv, cv::Scalar(160, 15,  40), cv::Scalar(180, 220, 255), mask2);
    cv::bitwise_or(mask1, mask2, skinMask);

    double skinRatio = (double)cv::countNonZero(skinMask) / (double)skinMask.total();
    // At least 10 % of crop should be skin-coloured (was 15 %, too strict for finger crops)
    return std::clamp((float)(skinRatio / 0.10), 0.0f, 1.0f);
}

LivenessResult LivenessDetector::detect(const cv::Mat& image, float handConfidence) {
    LivenessResult result{};

    cv::Mat gray, bgr;
    if (image.channels() == 1) {
        gray = image.clone();
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image.clone();
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    // Layer 2 (hard gate): glare
    float glareScore = checkGlare(gray);
    result.glareDetected = (glareScore <= 0.0f);
    if (result.glareDetected) {
        result.confidence = 0.1f;
        result.isLive     = false;
        return result;
    }

    // Layer 3: texture
    result.textureScore = computeTextureScore(gray);

    // Layer 4: skin colour
    result.skinScore = computeSkinScore(bgr);

    // Combine scores — start from the actual MediaPipe hand detection confidence
    // (mirrors reference: check_liveness(gray, bgr, hand_confidence) in server.py)
    // Penalty thresholds at 0.4; isLive threshold at 0.50
    float confidence = std::clamp(handConfidence, 0.0f, 1.0f);
    if (result.textureScore < 0.4f) confidence *= 0.65f;
    if (result.skinScore    < 0.4f) confidence *= 0.75f;

    result.confidence = std::clamp(confidence, 0.0f, 1.0f);
    result.isLive     = result.confidence >= 0.50f;
    return result;
}

} // namespace fingerprint
