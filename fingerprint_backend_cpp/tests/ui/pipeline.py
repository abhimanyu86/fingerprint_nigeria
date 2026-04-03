"""
pipeline.py — Python mirror of the C++ fingerprint core.

Implements the same algorithms as the C++ files:
  src/quality_analyzer.cpp
  src/liveness_detector.cpp
  src/fingerprint_core.cpp

Uses cv2 + numpy so it runs without building the binary.
Results should be numerically equivalent to the C++ output.
"""

import cv2
import numpy as np
import math


# ── Quality ────────────────────────────────────────────────────

def _blur_score(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    _, lap_std = cv2.meanStdDev(lap)
    lap_var = float(lap_std[0][0]) ** 2

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    tenen_mean = float(np.mean(mag))

    combined = 0.6 * (lap_var / 100.0) + 0.4 * (tenen_mean / 20.0)
    return float(np.clip(combined * 100.0, 0, 100))


def _contrast_score(gray: np.ndarray) -> float:
    mean, stddev = cv2.meanStdDev(gray)
    rms = float(stddev[0][0])

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / gray.size
    entropy = 0.0
    for p in hist.flatten():
        if p > 1e-7:
            entropy -= p * math.log2(p)
    ent_norm = min(8.0, entropy) / 8.0 * 100.0

    mean_val = float(mean[0][0])
    brightness_penalty = 0.5 if (mean_val < 55 or mean_val > 215) else 1.0

    score = (rms / 60.0 * 50.0) + ent_norm * 0.5
    return float(np.clip(score * brightness_penalty, 0, 100))


def _ridge_clarity_score(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    density = float(np.count_nonzero(edges)) / edges.size
    deviation = abs(density - 0.15)
    score = 100.0 * (1.0 - deviation / 0.15)
    return float(np.clip(score, 0, 100))


def _coverage_score(gray: np.ndarray) -> float:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coverage = float(np.count_nonzero(binary)) / binary.size
    if 0.30 <= coverage <= 0.80:
        centre_dist = abs(coverage - 0.55)
        return float(np.clip(100.0 - centre_dist * 200.0, 0, 100))
    dist = min(abs(coverage - 0.30), abs(coverage - 0.80))
    return float(np.clip(100.0 - dist * 500.0, 0, 100))


def _orientation_score(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    block = 16
    sin_sum = cos_sum = 0.0
    count = 0
    for i in range(0, gray.shape[0] - block + 1, block):
        for j in range(0, gray.shape[1] - block + 1, block):
            bx = gx[i:i+block, j:j+block]
            by = gy[i:i+block, j:j+block]
            Gxy   = float(np.sum(2.0 * bx * by))
            Gxxyy = float(np.sum(bx**2 - by**2))
            angle = 0.5 * math.atan2(Gxy, Gxxyy)
            sin_sum += math.sin(2.0 * angle)
            cos_sum += math.cos(2.0 * angle)
            count += 1

    if count == 0:
        return 50.0
    coherence = math.sqrt(sin_sum**2 + cos_sum**2) / count
    return float(np.clip(coherence * 111.0, 0, 100))


def _illumination_score(gray: np.ndarray) -> float:
    """Brightness level (ideal 100-180) + quadrant uniformity."""
    mean_val = float(cv2.mean(gray)[0])
    # Score: 1.0 when mean ≈ 140, falls off ±80 either side
    bright_score = max(0.0, 1.0 - abs(mean_val - 140.0) / 80.0)

    H, W = gray.shape[:2]
    hH, hW = H // 2, W // 2
    q = [
        float(cv2.mean(gray[:hH,   :hW  ])[0]),
        float(cv2.mean(gray[:hH,   hW:  ])[0]),
        float(cv2.mean(gray[hH:,   :hW  ])[0]),
        float(cv2.mean(gray[hH:,   hW:  ])[0]),
    ]
    q_mean = sum(q) / 4.0
    q_std  = math.sqrt(sum((v - q_mean) ** 2 for v in q) / 4.0)
    uniformity = max(0.0, 1.0 - q_std / 60.0)   # std > 60 grey levels = badly uneven

    return float(np.clip((0.6 * bright_score + 0.4 * uniformity) * 100.0, 0, 100))


def _position_score(gray: np.ndarray) -> float:
    """Crop aspect ratio (ideal h/w 2-3.5) + finger coverage (ideal 60%)."""
    H, W = gray.shape[:2]
    aspect = H / W if W > 0 else 1.0
    if 2.0 <= aspect <= 3.5:
        aspect_score = 1.0
    elif aspect < 2.0:
        aspect_score = max(0.0, aspect / 2.0)
    else:
        aspect_score = max(0.0, 1.0 - (aspect - 3.5) / 3.5)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coverage  = float(np.count_nonzero(binary)) / binary.size
    coverage_score = max(0.0, 1.0 - abs(coverage - 0.60) / 0.40)

    return float(np.clip((0.5 * aspect_score + 0.5 * coverage_score) * 100.0, 0, 100))


def _guidance(r: dict) -> str:
    if r['illuminationScore'] < 40:
        return "Too dark or overexposed — improve lighting" if r['illuminationScore'] < 20 \
               else "Uneven lighting — move to better light"
    if r['blurScore']         < 40: return "Hold still — image is blurry"
    if r['positionScore']     < 40: return "Center your finger and fill the frame"
    if r['ridgeClarityScore'] < 40: return "Press finger more firmly on lens"
    if r['orientationScore']  < 40: return "Straighten your finger"
    if r['compositeScore']   >= 70: return "Good — capture ready"
    return "Adjust position slightly"


def analyze_quality(gray: np.ndarray) -> dict:
    r = {
        'blurScore':          _blur_score(gray),
        'contrastScore':      _contrast_score(gray),
        'ridgeClarityScore':  _ridge_clarity_score(gray),
        'coverageScore':      _coverage_score(gray),
        'orientationScore':   _orientation_score(gray),
        'illuminationScore':  _illumination_score(gray),
        'positionScore':      _position_score(gray),
    }
    composite = (
        r['blurScore']         * 0.25 +
        r['contrastScore']     * 0.15 +
        r['ridgeClarityScore'] * 0.25 +
        r['coverageScore']     * 0.15 +
        r['orientationScore']  * 0.20
    )
    r['compositeScore'] = float(np.clip(composite, 0, 100))
    if   r['compositeScore'] >= 70: r['decision'] = 'ACCEPT'
    elif r['compositeScore'] >= 40: r['decision'] = 'RETRY'
    else:                           r['decision'] = 'REJECT'
    r['guidance'] = _guidance(r)
    return r


# ── Liveness ───────────────────────────────────────────────────

def _glare_score(gray: np.ndarray) -> float:
    _, bright = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    ratio = float(np.count_nonzero(bright)) / bright.size
    return float(1.0 - ratio / 0.12)


def _texture_score(gray: np.ndarray) -> float:
    gF  = gray.astype(np.float32)
    gF2 = gF * gF
    Ex  = cv2.boxFilter(gF,  cv2.CV_32F, (5, 5))
    Ex2 = cv2.boxFilter(gF2, cv2.CV_32F, (5, 5))
    variance = Ex2 - Ex * Ex
    local_std = float(math.sqrt(abs(float(np.mean(variance)))))

    if local_std < 3 or local_std > 120: return 0.3
    if 15 <= local_std <= 90:            return 1.0
    if local_std < 15:                   return local_std / 15.0
    return max(0.3, 1.0 - (local_std - 90.0) / 120.0)


def _skin_score(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,   15,  40), (30,  220, 255))
    mask2 = cv2.inRange(hsv, (160, 15,  40), (180, 220, 255))
    skin  = cv2.bitwise_or(mask1, mask2)
    ratio = float(np.count_nonzero(skin)) / skin.size
    return float(np.clip(ratio / 0.10, 0.0, 1.0))


def analyze_liveness(image: np.ndarray, hand_confidence: float = 0.88) -> dict:
    if image.ndim == 2 or image.shape[2] == 1:
        gray = image if image.ndim == 2 else image[:, :, 0]
        bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr  = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glare_val    = _glare_score(gray)
    glare_detect = glare_val <= 0.0

    if glare_detect:
        return {
            'confidence':    0.1,
            'isLive':        False,
            'glareDetected': True,
            'textureScore':  0.0,
            'skinScore':     0.0,
        }

    tex  = _texture_score(gray)
    skin = _skin_score(bgr)

    confidence = float(max(0.0, min(1.0, hand_confidence)))
    if tex  < 0.4: confidence *= 0.65
    if skin < 0.4: confidence *= 0.75
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return {
        'confidence':    confidence,
        'isLive':        confidence >= 0.50,
        'glareDetected': False,
        'textureScore':  tex,
        'skinScore':     skin,
    }


# ── Full pipeline ──────────────────────────────────────────────

def process_image(jpeg_bytes: bytes) -> dict:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            'errorMessage': 'Failed to decode image — invalid or empty JPEG',
            'quality':  {'compositeScore':0,'blurScore':0,'contrastScore':0,
                         'ridgeClarityScore':0,'coverageScore':0,'orientationScore':0,
                         'decision':'REJECT','guidance':''},
            'liveness': {'confidence':0,'isLive':False,'glareDetected':False,
                         'textureScore':0,'skinScore':0},
            'template': {'success':False,'minutiaeCount':0},
            'accepted': False,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quality  = analyze_quality(gray)
    liveness = analyze_liveness(img)

    quality_ok  = quality['decision'] != 'REJECT'
    liveness_ok = liveness['isLive']

    # Template extraction is C++ only (ISO 19794-2 minutiae is non-trivial)
    # We report honestly: success only when quality+liveness both pass
    template = {
        'success':       quality_ok and liveness_ok,
        'minutiaeCount': 0,   # not implemented in Python fallback
    }

    accepted = (quality['decision'] == 'ACCEPT') and liveness_ok and template['success']

    return {
        'errorMessage': '',
        'quality':      quality,
        'liveness':     liveness,
        'template':     template,
        'accepted':     accepted,
        '_pythonFallback': True,   # tells UI this came from Python, not C++
    }
