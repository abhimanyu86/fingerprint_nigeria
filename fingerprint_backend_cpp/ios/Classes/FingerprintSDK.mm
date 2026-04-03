// FingerprintSDK.mm — Objective-C++ bridge to the fingerprint C++ core
//
// MACRO COLLISION FIX
// OpenCV headers use identifiers (NO, YES, nil, Nil) that clash with ObjC
// runtime macros. Push/pop them around the C++ include.
#import "FingerprintSDK.h"

#pragma push_macro("NO")
#pragma push_macro("YES")
#pragma push_macro("nil")
#pragma push_macro("Nil")
#undef NO
#undef YES
#undef nil
#undef Nil

#include "fingerprint_core.h"

#pragma pop_macro("NO")
#pragma pop_macro("YES")
#pragma pop_macro("nil")
#pragma pop_macro("Nil")

// ---------------------------------------------------------------------------

@implementation FingerprintCaptureResult
@end

@implementation FingerCaptureResult
@end

// ── Module-level hand detector ──────────────────────────────────────────
static HandLandmarkDetector* gDetector = nil;
static dispatch_once_t       gDetectorToken;

// ── Internal helper: run C++ pipeline on raw bytes ─────────────────────
static FingerprintCaptureResult* runPipeline(NSData* jpegData,
                                             float handConfidence = 0.88f) {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(jpegData.bytes);
    std::vector<uint8_t> jpeg(bytes, bytes + jpegData.length);

    fingerprint::CaptureResult cpp = fingerprint::processFingerImage(jpeg, handConfidence);

    FingerprintCaptureResult* r = [[FingerprintCaptureResult alloc] init];
    r.qualityScore      = cpp.quality.compositeScore;
    r.blurScore         = cpp.quality.blurScore;
    r.contrastScore     = cpp.quality.contrastScore;
    r.ridgeClarityScore = cpp.quality.ridgeClarityScore;
    r.coverageScore     = cpp.quality.coverageScore;
    r.orientationScore  = cpp.quality.orientationScore;
    r.illuminationScore = cpp.quality.illuminationScore;
    r.positionScore     = cpp.quality.positionScore;
    r.guidance          = [NSString stringWithUTF8String:cpp.quality.guidance.c_str()];
    r.livenessScore     = cpp.liveness.confidence;
    r.isLive            = (BOOL)cpp.liveness.isLive;
    r.glareDetected     = (BOOL)cpp.liveness.glareDetected;
    r.textureScore      = cpp.liveness.textureScore;
    r.skinScore         = cpp.liveness.skinScore;
    r.accepted          = (BOOL)cpp.accepted;
    r.templateBase64    = cpp.templ.success
        ? [NSString stringWithUTF8String:cpp.templ.base64Template.c_str()]
        : nil;
    r.errorMessage      = cpp.errorMessage.empty()
        ? nil
        : [NSString stringWithUTF8String:cpp.errorMessage.c_str()];
    return r;
}

// ---------------------------------------------------------------------------

@implementation FingerprintSDK

// ── Init ────────────────────────────────────────────────────────────────

+ (BOOL)initWithModelPath:(NSString*)modelPath error:(NSError**)error {
    __block BOOL success = YES;
    dispatch_once(&gDetectorToken, ^{
        NSError* initErr = nil;
        gDetector = [[HandLandmarkDetector alloc] initWithModelPath:modelPath
                                                              error:&initErr];
        if (!gDetector) {
            if (error) *error = initErr;
            success = NO;
        }
    });
    return success;
}

// ── Full-frame processing ────────────────────────────────────────────────

+ (NSArray<FingerCaptureResult*>*)processFrame:(NSData*)jpegData {
    if (!gDetector) {
        NSLog(@"[FingerprintSDK] WARNING: initWithModelPath:error: not called. "
              @"Returning empty results.");
        return @[];
    }

    NSArray<DetectedFinger*>* detected = [gDetector detectInJpeg:jpegData];
    NSMutableArray<FingerCaptureResult*>* results =
        [NSMutableArray arrayWithCapacity:detected.count];

    for (DetectedFinger* df in detected) {
        FingerprintCaptureResult* pipeline = runPipeline(df.croppedJpeg, df.detectionScore);

        FingerCaptureResult* fcr  = [[FingerCaptureResult alloc] init];
        fcr.handedness            = df.handedness;
        fcr.fingerName            = df.fingerName;
        fcr.fingerId              = df.fingerId;
        fcr.detectionScore        = df.detectionScore;
        fcr.cropRect              = df.cropRect;
        fcr.pipeline              = pipeline;

        [results addObject:fcr];
    }

    return [results copy];
}

+ (NSArray<FingerCaptureResult*>*)processFrameAcceptedOnly:(NSData*)jpegData {
    NSArray<FingerCaptureResult*>* all = [self processFrame:jpegData];
    NSPredicate* pred = [NSPredicate predicateWithBlock:
        ^BOOL(FingerCaptureResult* r, NSDictionary* _) { return r.pipeline.accepted; }];
    return [all filteredArrayUsingPredicate:pred];
}

// ── Pre-cropped finger ───────────────────────────────────────────────────

+ (FingerprintCaptureResult*)processImage:(NSData*)jpegData {
    return runPipeline(jpegData);
}

// ── Matching ─────────────────────────────────────────────────────────────

+ (float)matchTemplate:(NSString*)templateA
          withTemplate:(NSString*)templateB {
    fingerprint::MatchResult res = fingerprint::Matcher::match(
        std::string(templateA.UTF8String),
        std::string(templateB.UTF8String));
    return res.score;
}

+ (BOOL)isMatch:(NSString*)templateA
   withTemplate:(NSString*)templateB {
    return [self matchTemplate:templateA withTemplate:templateB] >= 0.35f;
}

// ── Quality only ─────────────────────────────────────────────────────────

+ (float)qualityScoreForImage:(NSData*)jpegData {
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(jpegData.bytes);
    std::vector<uint8_t> jpeg(bytes, bytes + jpegData.length);
    cv::Mat img = fingerprint::decodeImage(jpeg);
    if (img.empty()) return 0.0f;
    fingerprint::QualityResult q = fingerprint::QualityAnalyzer::analyze(img);
    return q.compositeScore;
}

@end
