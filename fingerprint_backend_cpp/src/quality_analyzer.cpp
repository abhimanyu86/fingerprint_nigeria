#include "quality_analyzer.h"
#include <cmath>
#include <algorithm>

namespace fingerprint {

// ---------------------------------------------------------------------------
// Private metrics
// ---------------------------------------------------------------------------

float QualityAnalyzer::computeBlurScore(const cv::Mat& gray) {
    // Laplacian variance
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    double lapVar = stddev[0] * stddev[0];

    // Tenengrad (Sobel energy)
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_64F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_64F, 0, 1, 3);
    cv::Mat mag;
    cv::sqrt(gx.mul(gx) + gy.mul(gy), mag);
    double tenenMean = cv::mean(mag)[0];

    // Blend: Laplacian variance dominates
    double combined = 0.6 * (lapVar / 100.0) + 0.4 * (tenenMean / 20.0);
    return std::clamp((float)(combined * 100.0), 0.0f, 100.0f);
}

float QualityAnalyzer::computeContrastScore(const cv::Mat& gray) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    double rms = stddev[0];

    // Histogram entropy
    cv::Mat hist;
    float range[] = {0, 256};
    const float* histRange = range;
    int histSize = 256;
    cv::calcHist(&gray, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);
    hist /= (float)gray.total();
    float entropy = 0;
    for (int i = 0; i < 256; i++) {
        float p = hist.at<float>(i);
        if (p > 1e-7f) entropy -= p * std::log2(p);
    }
    float entNorm = std::min(8.0f, entropy) / 8.0f * 100.0f;

    // Penalty for extreme brightness (dark < 55, glare > 215)
    float brightnessPenalty = (mean[0] < 55 || mean[0] > 215) ? 0.5f : 1.0f;

    float score = (float)(rms / 60.0 * 50.0) + entNorm * 0.5f;
    return std::clamp(score * brightnessPenalty, 0.0f, 100.0f);
}

float QualityAnalyzer::computeRidgeClarityScore(const cv::Mat& gray) {
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    double density = (double)cv::countNonZero(edges) / (double)edges.total();

    // Ideal density ~15 %
    double deviation = std::abs(density - 0.15);
    float score = (float)(100.0 * (1.0 - deviation / 0.15));
    return std::clamp(score, 0.0f, 100.0f);
}

float QualityAnalyzer::computeCoverageScore(const cv::Mat& gray) {
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    double coverage = (double)cv::countNonZero(binary) / (double)binary.total();

    // Ideal 30-80 %; penalise outside that band
    if (coverage >= 0.30 && coverage <= 0.80) {
        float centreDist = (float)std::abs(coverage - 0.55);
        return std::clamp(100.0f - centreDist * 200.0f, 0.0f, 100.0f);
    }
    float dist = (float)std::min(std::abs(coverage - 0.30), std::abs(coverage - 0.80));
    return std::clamp(100.0f - dist * 500.0f, 0.0f, 100.0f);
}

float QualityAnalyzer::computeOrientationScore(const cv::Mat& gray) {
    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

    const int blockSize = 16;
    float sinSum = 0, cosSum = 0;
    int   count  = 0;

    for (int i = 0; i + blockSize <= gray.rows; i += blockSize) {
        for (int j = 0; j + blockSize <= gray.cols; j += blockSize) {
            float Gxy = 0, Gxx_yy = 0;
            for (int y = i; y < i + blockSize; y++) {
                for (int x = j; x < j + blockSize; x++) {
                    float dx = gx.at<float>(y, x);
                    float dy = gy.at<float>(y, x);
                    Gxy    += 2.0f * dx * dy;
                    Gxx_yy += dx * dx - dy * dy;
                }
            }
            float angle = 0.5f * std::atan2(Gxy, Gxx_yy);
            sinSum += std::sin(2.0f * angle);
            cosSum += std::cos(2.0f * angle);
            count++;
        }
    }

    if (count == 0) return 50.0f;

    float coherence = std::sqrt(sinSum * sinSum + cosSum * cosSum) / count;
    // Real fingerprint coherence ~0.6-0.9 → score 67-100
    return std::clamp(coherence * 111.0f, 0.0f, 100.0f);
}

float QualityAnalyzer::computeIlluminationScore(const cv::Mat& gray) {
    // Brightness: ideal mean 100-180 (well-lit, not overexposed)
    cv::Scalar meanVal = cv::mean(gray);
    double brightness  = meanVal[0];
    const double idealCenter = 140.0;
    const double idealRange  = 80.0;   // soft falloff ±80 around 140
    double brightScore = std::max(0.0, 1.0 - std::abs(brightness - idealCenter) / idealRange);

    // Uniformity: quadrant brightness variance — uneven lighting = low score
    int H = gray.rows, W = gray.cols;
    double q[4] = {
        cv::mean(gray(cv::Rect(0,   0,   W/2,   H/2  )))[0],
        cv::mean(gray(cv::Rect(W/2, 0,   W-W/2, H/2  )))[0],
        cv::mean(gray(cv::Rect(0,   H/2, W/2,   H-H/2)))[0],
        cv::mean(gray(cv::Rect(W/2, H/2, W-W/2, H-H/2)))[0],
    };
    double qMean = (q[0]+q[1]+q[2]+q[3]) / 4.0;
    double qVar  = 0.0;
    for (int i = 0; i < 4; i++) qVar += (q[i]-qMean)*(q[i]-qMean);
    qVar /= 4.0;
    // std dev > 60 grey levels across quadrants = badly uneven
    double uniformity = std::max(0.0, 1.0 - std::sqrt(qVar) / 60.0);

    return std::clamp((float)((0.6 * brightScore + 0.4 * uniformity) * 100.0), 0.0f, 100.0f);
}

float QualityAnalyzer::computePositionScore(const cv::Mat& gray) {
    // Aspect ratio: a finger crop should be portrait, h/w ≈ 2.0–3.5
    double aspectRatio = (double)gray.rows / (double)gray.cols;
    double aspectScore;
    if (aspectRatio >= 2.0 && aspectRatio <= 3.5) {
        aspectScore = 1.0;
    } else if (aspectRatio < 2.0) {
        aspectScore = std::max(0.0, aspectRatio / 2.0);
    } else {
        aspectScore = std::max(0.0, 1.0 - (aspectRatio - 3.5) / 3.5);
    }

    // Foreground coverage: how much of the crop the finger occupies (ideal 40-80%)
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    double coverage = (double)cv::countNonZero(binary) / (double)binary.total();
    // Score peaks at 60% coverage, falls off symmetrically
    double coverageScore = std::max(0.0, 1.0 - std::abs(coverage - 0.60) / 0.40);

    return std::clamp((float)((0.5 * aspectScore + 0.5 * coverageScore) * 100.0), 0.0f, 100.0f);
}

std::string QualityAnalyzer::generateGuidance(const QualityResult& r) {
    // Priority: illumination > blur > position > ridges > orientation
    if (r.illuminationScore < 40.0f) {
        return r.illuminationScore < 20.0f
            ? "Too dark or overexposed — improve lighting"
            : "Uneven lighting — move to better light";
    }
    if (r.blurScore         < 40.0f) return "Hold still — image is blurry";
    if (r.positionScore     < 40.0f) return "Center your finger and fill the frame";
    if (r.ridgeClarityScore < 40.0f) return "Press finger more firmly on lens";
    if (r.orientationScore  < 40.0f) return "Straighten your finger";
    if (r.compositeScore    >= 70.0f) return "Good — capture ready";
    return "Adjust position slightly";
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

QualityResult QualityAnalyzer::analyze(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() > 1)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    QualityResult r;
    r.blurScore         = computeBlurScore(gray);
    r.contrastScore     = computeContrastScore(gray);
    r.ridgeClarityScore = computeRidgeClarityScore(gray);
    r.coverageScore     = computeCoverageScore(gray);
    r.orientationScore  = computeOrientationScore(gray);
    r.illuminationScore = computeIlluminationScore(gray);
    r.positionScore     = computePositionScore(gray);

    r.compositeScore =
        r.blurScore         * 0.25f +
        r.contrastScore     * 0.15f +
        r.ridgeClarityScore * 0.25f +
        r.coverageScore     * 0.15f +
        r.orientationScore  * 0.20f;

    r.compositeScore = std::clamp(r.compositeScore, 0.0f, 100.0f);

    if      (r.compositeScore >= 70.0f) r.decision = QualityDecision::ACCEPT;
    else if (r.compositeScore >= 40.0f) r.decision = QualityDecision::RETRY;
    else                                r.decision = QualityDecision::REJECT;

    r.guidance = generateGuidance(r);
    return r;
}

} // namespace fingerprint
