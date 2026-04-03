#import <Foundation/Foundation.h>
#import "HandLandmarkDetector.h"

NS_ASSUME_NONNULL_BEGIN

// ── Pipeline result for one finger ─────────────────────────────────────

/** Full pipeline result for a single finger crop. */
@interface FingerprintCaptureResult : NSObject

// Quality
@property (nonatomic) float qualityScore;
@property (nonatomic) float blurScore;
@property (nonatomic) float contrastScore;
@property (nonatomic) float ridgeClarityScore;
@property (nonatomic) float coverageScore;
@property (nonatomic) float orientationScore;
// Primary user-facing scores (shown per finger in UI)
@property (nonatomic) float illuminationScore;  ///< brightness + uniformity  0-100
@property (nonatomic) float positionScore;      ///< crop aspect ratio + coverage  0-100
@property (nonatomic, strong) NSString* guidance;

// Liveness
@property (nonatomic) float livenessScore;
@property (nonatomic) BOOL  isLive;
@property (nonatomic) BOOL  glareDetected;
@property (nonatomic) float textureScore;
@property (nonatomic) float skinScore;

// Overall
@property (nonatomic) BOOL  accepted;
@property (nonatomic, strong, nullable) NSString* templateBase64;
@property (nonatomic, strong, nullable) NSString* errorMessage;

@end


// ── Per-finger result (when processing a full frame) ───────────────────

/**
 * Result from processFrame: one entry per detected finger.
 * Combines MediaPipe detection metadata with the C++ pipeline result.
 */
@interface FingerCaptureResult : NSObject

/// "Left" or "Right"
@property (nonatomic, strong) NSString* handedness;
/// "Thumb" | "Index" | "Middle" | "Ring" | "Pinky"
@property (nonatomic, strong) NSString* fingerName;
/// ISO/ANSI finger position: Right Thumb=1…Pinky=5, Left Thumb=6…Pinky=10
@property (nonatomic) NSInteger fingerId;
/// MediaPipe detection confidence 0.0–1.0
@property (nonatomic) float detectionScore;
/// Bounding rect of the crop in the original image
@property (nonatomic) CGRect cropRect;
/// Full pipeline result for this finger
@property (nonatomic, strong) FingerprintCaptureResult* pipeline;

@end


// ── Main SDK class ─────────────────────────────────────────────────────

/** Main SDK entry point. All methods are thread-safe. */
@interface FingerprintSDK : NSObject

// ── Initialization ───────────────────────────────────────────────────

/**
 * Initialize MediaPipe hand detection.
 * Must be called once before processFrame:.
 *
 * @param modelPath  Path to hand_landmarker.task in your app bundle:
 *                   [[NSBundle mainBundle] pathForResource:@"hand_landmarker" ofType:@"task"]
 * @param error      Set on failure.
 * @return           YES on success.
 */
+ (BOOL)initWithModelPath:(NSString*)modelPath error:(NSError**)error;

// ── Full-frame API (hand detection + crop + pipeline) ────────────────

/**
 * Process a full camera frame.
 *
 * 1. MediaPipe detects up to 2 hands (10 fingers).
 * 2. Each fingertip region is cropped automatically.
 * 3. The fingerprint pipeline (quality → liveness → template) runs on each crop.
 *
 * @param jpegData  Full-resolution JPEG from AVCaptureOutput.
 * @return          Array of FingerCaptureResult, one per detected finger (0–10 items).
 */
+ (NSArray<FingerCaptureResult*>*)processFrame:(NSData*)jpegData;

/**
 * Convenience: like processFrame:, but only returns results where accepted == YES.
 */
+ (NSArray<FingerCaptureResult*>*)processFrameAcceptedOnly:(NSData*)jpegData;

// ── Pre-cropped finger API ───────────────────────────────────────────

/**
 * Run the full pipeline on a pre-cropped JPEG finger image.
 */
+ (FingerprintCaptureResult*)processImage:(NSData*)jpegData;

/**
 * Match two ISO 19794-2 base64-encoded templates.
 * @return Score 0.0–1.0 (threshold: 0.35)
 */
+ (float)matchTemplate:(NSString*)templateA
          withTemplate:(NSString*)templateB;

/** Convenience: returns YES when score >= 0.35. */
+ (BOOL)isMatch:(NSString*)templateA
   withTemplate:(NSString*)templateB;

/** Quick quality score only (skips liveness + template encoding). */
+ (float)qualityScoreForImage:(NSData*)jpegData;

@end

NS_ASSUME_NONNULL_END
