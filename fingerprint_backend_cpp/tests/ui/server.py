"""
Fingerprint SDK — Web Test UI Server
Mirrors fingerprint_backend reference implementation.

Run:  python tests/ui/server.py
UI:   http://localhost:5000
"""
import os, sys, json, tempfile, subprocess, base64, io
from pathlib import Path
import numpy as np
import cv2

# MediaPipe — server-side hand detection (no CDN dependency)
import mediapipe as mp
_mp_hands   = mp.solutions.hands
_HANDS      = _mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
)

from flask import Flask, request, jsonify, send_file
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as py_pipeline

# ── C++ binary (optional) ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
BINARY_CANDIDATES = [
    ROOT / "build" / "Release" / "fp_test.exe",
    ROOT / "build" / "fp_test.exe",
    ROOT / "build" / "Release" / "fp_test",
    ROOT / "build" / "fp_test",
]
def _find_binary():
    ep = os.environ.get("FP_TEST_BIN")
    if ep and Path(ep).exists(): return str(Path(ep))
    for p in BINARY_CANDIDATES:
        if p.exists(): return str(p)
    return None
FP_BIN = _find_binary()

# ── Finger definitions (matches reference) ────────────────────────────
_FINGER_DEFS = {
    "INDEX":  [5, 6, 7, 8],
    "MIDDLE": [9, 10, 11, 12],
    "RING":   [13, 14, 15, 16],
    "LITTLE": [17, 18, 19, 20],
}
_BBOX_PAD = 45

# ── Hand detection (mirrors hand_detector.py reference) ───────────────

def _is_finger_extended(lm, ids):
    return lm.landmark[ids[-1]].y < lm.landmark[ids[0]].y - 0.04

def _analyze_geometry(lm):
    wrist = lm.landmark[0]
    m_tip = lm.landmark[12]
    span  = abs(m_tip.y - wrist.y)
    if span < 0.30: return "Move your hand closer to the camera"
    if span > 0.72: return "Move your hand back from the camera"

    palm_xs = [lm.landmark[i].x for i in [0,5,9,13,17]]
    palm_ys = [lm.landmark[i].y for i in [0,5,9,13,17]]
    px, py  = float(np.mean(palm_xs)), float(np.mean(palm_ys))
    if px < 0.20: return "Move your hand to the right"
    if px > 0.80: return "Move your hand to the left"
    if py < 0.12: return "Move your hand down"
    if py > 0.88: return "Move your hand up"

    tilt = abs(lm.landmark[5].y - lm.landmark[17].y)
    if tilt > 0.10: return "Keep your hand level — don't tilt it sideways"

    spread = abs(lm.landmark[8].x - lm.landmark[20].x)
    if spread < 0.12: return "Spread your fingers apart"
    return None

def _landmarks_to_bbox(lm, ids, W, H):
    xs = [lm.landmark[i].x * W for i in ids]
    ys = [lm.landmark[i].y * H for i in ids]
    x1 = max(0, int(min(xs)) - _BBOX_PAD)
    x2 = min(W, int(max(xs)) + _BBOX_PAD)
    y1 = max(0, int(min(ys)) - _BBOX_PAD)
    y2 = min(H, int(max(ys)) + _BBOX_PAD)
    w, h = x2-x1, y2-y1
    if w < 40 or h < 40: return None
    return (x1, y1, w, h)

def detect_all_fingers(bgr: np.ndarray):
    """Returns (hand_detected, confidence, guidance, fingers_dict, handedness)"""
    H, W = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res  = _HANDS.process(rgb)

    empty = {k: {"detected": False, "bbox": None, "crop": None}
             for k in _FINGER_DEFS}

    if not res.multi_hand_landmarks:
        return False, 0.0, "Place your hand flat in front of the camera", empty, None

    lm   = res.multi_hand_landmarks[0]
    hand_side = None
    conf = 0.88
    if res.multi_handedness:
        cls       = res.multi_handedness[0].classification[0]
        hand_side = cls.label
        conf      = cls.score

    geo_guidance = _analyze_geometry(lm)

    fingers = {}
    for name, ids in _FINGER_DEFS.items():
        if not _is_finger_extended(lm, ids):
            fingers[name] = {"detected": False, "bbox": None, "crop": None}
            continue
        bbox = _landmarks_to_bbox(lm, ids, W, H)
        if bbox is None:
            fingers[name] = {"detected": False, "bbox": None, "crop": None}
            continue
        x, y, w, h = bbox
        crop = bgr[y:y+h, x:x+w]
        fingers[name] = {"detected": True, "bbox": bbox, "crop": crop}

    if geo_guidance:
        guidance = geo_guidance
    else:
        missing  = [n.capitalize() for n, f in fingers.items() if not f["detected"]]
        if not missing:
            guidance = None
        else:
            guidance = "Raise your " + ", ".join(missing) + " finger" + ("s" if len(missing)>1 else "")

    detected_count = sum(1 for f in fingers.values() if f["detected"])
    if detected_count < 4:
        conf = 0.55 + 0.1 * detected_count

    return True, conf, guidance, fingers, hand_side

# ── Liveness check (mirrors liveness_detector.py reference) ──────────

_GLARE_MAX   = 0.12
_LBP_MIN     = 18.0
_SKIN_MIN    = 0.15

def _lbp_variance(gray):
    f   = gray.astype(np.float32)
    mu  = cv2.blur(f, (5,5))
    mu2 = cv2.blur(f*f, (5,5))
    return float(np.mean(np.sqrt(np.clip(mu2 - mu*mu, 0, None))))

def _skin_ratio(bgr):
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1    = cv2.inRange(hsv, (0,20,50),  (25,170,255))
    m2    = cv2.inRange(hsv, (165,20,50),(180,170,255))
    return float(cv2.countNonZero(cv2.bitwise_or(m1,m2))) / (bgr.shape[0]*bgr.shape[1])

def check_liveness(gray, bgr, hand_confidence):
    _, bright = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    glare     = cv2.countNonZero(bright) / float(gray.shape[0]*gray.shape[1])
    if glare > _GLARE_MAX:
        return False, "Screen replay or excessive glare", 0.1

    lbp_ok  = _lbp_variance(gray)  >= _LBP_MIN
    skin_ok = _skin_ratio(bgr)     >= _SKIN_MIN

    conf = hand_confidence
    conf *= (1.0 - 0.08*(glare/_GLARE_MAX))
    if not lbp_ok:  conf *= 0.75
    if not skin_ok: conf *= 0.80
    conf = float(np.clip(conf, 0.0, 1.0))

    if not lbp_ok and not skin_ok:
        return False, "Low texture and no skin tone detected", conf
    return True, None, conf

# ── App ───────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)

@app.route("/")
def index():
    return send_file(str(Path(__file__).parent / "index.html"))

@app.route("/status")
def status():
    return jsonify({
        "binary":      FP_BIN,
        "binaryFound": FP_BIN is not None,
        "mode":        "cpp" if FP_BIN else "python",
    })

# ── /analyze  (fast, no template — for live polling) ─────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    hand_side = request.form.get("hand", "RIGHT").upper()
    img_file  = request.files.get("image")
    if not img_file:
        return jsonify({"error": "no image"}), 400

    arr = np.frombuffer(img_file.read(), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"hand_detected": False, "guidance": "Failed to decode image", "fingers": []})

    H, W    = bgr.shape[:2]
    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hand_ok, hand_conf, guidance, fingers, detected_side = detect_all_fingers(bgr)

    liveness_passed, liveness_reason, liveness_conf = (False, "No hand", 0.0)
    if hand_ok:
        liveness_passed, liveness_reason, liveness_conf = check_liveness(gray, bgr, hand_conf)

    fingers_out = []
    for fname in ["INDEX", "MIDDLE", "RING", "LITTLE"]:
        finger_id = f"{hand_side}_{fname}"
        fc = fingers.get(fname, {})
        if not fc.get("detected") or fc.get("crop") is None:
            fingers_out.append({
                "finger_id": finger_id, "detected": False,
                "quality_score": 0, "blur_score": 0,
                "illumination_score": 0, "position_score": 0,
                "contrast_score": 0, "ridge_score": 0,
                "coverage_score": 0, "orientation_score": 0,
                "liveness": False, "liveness_conf": 0,
                "guidance": "Finger not visible", "bbox_pct": None
            })
            continue

        crop_gray = cv2.cvtColor(fc["crop"], cv2.COLOR_BGR2GRAY)
        q = py_pipeline.analyze_quality(crop_gray)
        # use actual MediaPipe hand confidence as liveness base
        liveness_local = py_pipeline.analyze_liveness(fc["crop"], hand_confidence=hand_conf if hand_ok else 0.0)

        bbox = fc["bbox"]
        bbox_pct = None
        if bbox:
            x, y, w, h = bbox
            bbox_pct = {"x": x/W, "y": y/H, "w": w/W, "h": h/H}

        # Crop thumbnail as base64
        _, buf = cv2.imencode(".jpg", fc["crop"], [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumb_b64 = base64.b64encode(buf).decode()

        fingers_out.append({
            "finger_id":          finger_id,
            "detected":           True,
            "quality_score":      round(q["compositeScore"], 1),
            "blur_score":         round(q["blurScore"], 1),
            "illumination_score": round(q["illuminationScore"], 1),
            "position_score":     round(q["positionScore"], 1),
            "contrast_score":     round(q["contrastScore"], 1),
            "ridge_score":        round(q["ridgeClarityScore"], 1),
            "coverage_score":     round(q["coverageScore"], 1),
            "orientation_score":  round(q["orientationScore"], 1),
            "guidance":           q["guidance"],
            "decision":           q["decision"],
            "liveness":           liveness_local["isLive"],
            "liveness_conf":      round(liveness_local["confidence"], 3),
            "bbox_pct":           bbox_pct,
            "thumb_b64":          thumb_b64,
        })

    return jsonify({
        "hand_detected":   hand_ok,
        "hand_confidence": round(hand_conf, 3),
        "hand_side":       detected_side,
        "guidance":        guidance,
        "liveness_passed": liveness_passed,
        "liveness_conf":   round(liveness_conf, 3),
        "fingers":         fingers_out,
    })

# ── /capture/single  (full pipeline, with template) ──────────────────
@app.route("/capture/single", methods=["POST"])
def capture_single():
    img_file = request.files.get("image")
    if not img_file:
        return jsonify({"errorMessage": "no image"}), 400
    result = py_pipeline.process_image(img_file.read())
    return jsonify(result)

# ── /match ────────────────────────────────────────────────────────────
@app.route("/match", methods=["POST"])
def match():
    img1 = request.files.get("image1")
    img2 = request.files.get("image2")
    if not img1 or not img2:
        return jsonify({"errorMessage": "need image1 and image2"}), 400
    r1 = py_pipeline.process_image(img1.read())
    r2 = py_pipeline.process_image(img2.read())
    return jsonify({
        "image1": r1, "image2": r2,
        "match": {"possible": False, "score": 0, "isMatch": False,
                  "_note": "Template matching requires C++ binary"}
    })

if __name__ == "__main__":
    mode = "C++ binary" if FP_BIN else "Python/cv2 (mediapipe server-side)"
    print("=" * 55)
    print("  Fingerprint SDK — Web Test UI")
    print(f"  Mode : {mode}")
    print(f"  UI   : http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False)
