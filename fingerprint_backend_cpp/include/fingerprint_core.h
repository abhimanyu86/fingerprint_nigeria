#pragma once
#include "image_processor.h"
#include "quality_analyzer.h"
#include "liveness_detector.h"
#include "template_encoder.h"
#include "matcher.h"
#include <vector>
#include <cstdint>

namespace fingerprint {

// Full single-finger pipeline result
struct CaptureResult {
    QualityResult  quality;
    LivenessResult liveness;
    TemplateResult templ;
    bool           accepted;      // quality ACCEPT + liveness pass + template success
    std::string    errorMessage;  // non-empty when pipeline failed early (e.g. bad JPEG)
};

// Run full pipeline: decode JPEG → quality → liveness → template
// handConfidence: MediaPipe hand detection score forwarded to liveness detector.
// Defaults to 0.88 when not available (e.g. CLI test runner).
CaptureResult processFingerImage(const std::vector<uint8_t>& jpegBytes,
                                 float handConfidence = 0.88f);

// Helpers
cv::Mat              decodeImage(const std::vector<uint8_t>& jpegBytes);
std::vector<uint8_t> encodeImage(const cv::Mat& image, int quality = 90);

} // namespace fingerprint
