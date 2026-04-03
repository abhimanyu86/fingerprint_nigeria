#import "HandLandmarkDetector.h"
#import <MediaPipeTasksVision/MediaPipeTasksVision.h>

// ── 4-finger definitions (INDEX, MIDDLE, RING, LITTLE) ─────────────────
// Each row: { MCP, PIP, DIP, TIP } landmark indices
static const int kFingerLM[4][4] = {
    {5,  6,  7,  8},   // INDEX
    {9,  10, 11, 12},  // MIDDLE
    {13, 14, 15, 16},  // RING
    {17, 18, 19, 20},  // LITTLE
};
static NSString* const kFingerNames[4] = {
    @"Index", @"Middle", @"Ring", @"Little"
};
// ISO/ANSI: Right INDEX=2…LITTLE=5 | Left INDEX=7…LITTLE=10
static const int kRightIds[4] = {2, 3, 4, 5};
static const int kLeftIds[4]  = {7, 8, 9, 10};

static const CGFloat kBboxPad  = 45.0;
static const CGFloat kMinCrop  = 40.0;

// ── DetectedFinger ─────────────────────────────────────────────────────
@implementation DetectedFinger
@end

// ── HandLandmarkDetector ───────────────────────────────────────────────
@implementation HandLandmarkDetector {
    MPPHandLandmarker* _landmarker;
}

- (nullable instancetype)initWithModelPath:(NSString*)modelPath
                                     error:(NSError**)error {
    self = [super init];
    if (!self) return nil;

    MPPHandLandmarkerOptions* opts = [[MPPHandLandmarkerOptions alloc] init];
    opts.baseOptions.modelAssetPath = modelPath;
    opts.runningMode                = MPPRunningModeImage;
    opts.numHands                   = 1;
    opts.minHandDetectionConfidence = 0.45f;
    opts.minHandPresenceConfidence  = 0.45f;
    opts.minTrackingConfidence      = 0.45f;

    _landmarker = [[MPPHandLandmarker alloc] initWithOptions:opts error:error];
    return (_landmarker != nil) ? self : nil;
}

- (NSArray<DetectedFinger*>*)detectInJpeg:(NSData*)jpegData {
    NSMutableArray<DetectedFinger*>* results = [NSMutableArray array];

    UIImage* image = [UIImage imageWithData:jpegData];
    if (!image) return results;

    MPPImage* mpImage = [[MPPImage alloc] initWithUIImage:image error:nil];
    if (!mpImage) return results;

    NSError* err = nil;
    MPPHandLandmarkerResult* detection = [_landmarker detectInImage:mpImage error:&err];
    if (!detection || err) return results;

    CGFloat W = image.size.width;
    CGFloat H = image.size.height;

    NSUInteger handCount = detection.landmarks.count;
    for (NSUInteger h = 0; h < handCount; h++) {
        NSArray<MPPNormalizedLandmark*>* landmarks = detection.landmarks[h];

        // Handedness — MediaPipe mirrors front camera labels, flip to match reality
        NSString* handedness = @"Right";
        float detScore = 0.88f;
        if (h < detection.handednesses.count &&
            detection.handednesses[h].count > 0) {
            MPPCategory* cat = detection.handednesses[h][0];
            handedness = [cat.categoryName isEqualToString:@"Left"] ? @"Right" : @"Left";
            detScore   = cat.score;
        }

        const int* fingerIds = [handedness isEqualToString:@"Right"] ? kRightIds : kLeftIds;

        for (int f = 0; f < 4; f++) {
            const int* ids = kFingerLM[f];
            MPPNormalizedLandmark* mcp = landmarks[ids[0]];
            MPPNormalizedLandmark* tip = landmarks[ids[3]];

            // Extension check: tip.y < mcp.y - 0.04 (normalised coords, y↓)
            if (!(tip.y < mcp.y - 0.04f)) continue;

            // Full-finger bounding box: span all 4 landmarks in this finger
            CGFloat minX = CGFLOAT_MAX, maxX = -CGFLOAT_MAX;
            CGFloat minY = CGFLOAT_MAX, maxY = -CGFLOAT_MAX;
            for (int k = 0; k < 4; k++) {
                MPPNormalizedLandmark* lm = landmarks[ids[k]];
                if (lm.x < minX) minX = lm.x;
                if (lm.x > maxX) maxX = lm.x;
                if (lm.y < minY) minY = lm.y;
                if (lm.y > maxY) maxY = lm.y;
            }

            CGFloat x1 = MAX(0, minX * W - kBboxPad);
            CGFloat y1 = MAX(0, minY * H - kBboxPad);
            CGFloat x2 = MIN(W, maxX * W + kBboxPad);
            CGFloat y2 = MIN(H, maxY * H + kBboxPad);
            CGFloat cw = x2 - x1;
            CGFloat ch = y2 - y1;

            if (cw < kMinCrop || ch < kMinCrop) continue;

            CGRect cropRect  = CGRectMake(x1, y1, cw, ch);
            CGImageRef cgCrop = CGImageCreateWithImageInRect(image.CGImage, cropRect);
            UIImage* cropped  = [UIImage imageWithCGImage:cgCrop
                                                    scale:image.scale
                                              orientation:image.imageOrientation];
            CGImageRelease(cgCrop);

            NSData* jpeg = UIImageJPEGRepresentation(cropped, 0.92f);
            if (!jpeg) continue;

            DetectedFinger* df = [[DetectedFinger alloc] init];
            df.handedness      = handedness;
            df.fingerName      = kFingerNames[f];
            df.fingerId        = fingerIds[f];
            df.detectionScore  = detScore;
            df.croppedJpeg     = jpeg;
            df.cropRect        = cropRect;

            [results addObject:df];
        }
    }

    return [results copy];
}

@end
