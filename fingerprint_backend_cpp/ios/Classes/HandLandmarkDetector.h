#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

/**
 * One detected finger with its cropped image region.
 * Produced by HandLandmarkDetector.
 */
@interface DetectedFinger : NSObject

/// "Left" or "Right" (corrected for selfie/rear camera)
@property (nonatomic, copy)   NSString* handedness;

/// "Index" | "Middle" | "Ring" | "Little"
@property (nonatomic, copy)   NSString* fingerName;

/// ISO/ANSI position: Right INDEX=2…LITTLE=5 | Left INDEX=7…LITTLE=10
@property (nonatomic)         NSInteger fingerId;

/// MediaPipe hand detection confidence 0.0–1.0
@property (nonatomic)         float     detectionScore;

/// Cropped JPEG bytes ready for FingerprintSDK.processImage:
@property (nonatomic, strong) NSData*   croppedJpeg;

/// Bounding rect of the crop in the original image (points)
@property (nonatomic)         CGRect    cropRect;

@end


/**
 * Wraps MediaPipe Hand Landmarker to detect hand landmarks in a full
 * camera frame and return cropped finger-tip regions.
 *
 * Model file: hand_landmarker.task must be included in the app bundle.
 * Download: https://storage.googleapis.com/mediapipe-models/hand_landmarker/
 *           hand_landmarker/float16/latest/hand_landmarker.task
 */
@interface HandLandmarkDetector : NSObject

/**
 * Initialise the detector.
 *
 * @param modelPath  Absolute path to hand_landmarker.task.
 *                   Use [[NSBundle mainBundle] pathForResource:@"hand_landmarker"
 *                                                      ofType:@"task"]
 * @param error      Set on failure (model not found, MediaPipe init error, etc.)
 */
- (nullable instancetype)initWithModelPath:(NSString*)modelPath
                                     error:(NSError**)error;

/**
 * Detect all finger-tip regions in a JPEG frame.
 *
 * @param jpegData   Full-resolution JPEG (e.g. from AVCaptureOutput)
 * @return           Array of DetectedFinger (0–10 items). Empty if no hand found.
 */
- (NSArray<DetectedFinger*>*)detectInJpeg:(NSData*)jpegData;

@end

NS_ASSUME_NONNULL_END
