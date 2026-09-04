#!/usr/bin/env python3
"""
Face detection + encoding.

YuNet locates faces; SFace turns one into a 128-d embedding. SFace's own
alignCrop() is used for the recognition path because it produces exactly the
112x112 alignment the recognition model was trained on -- the hand-rolled
2-point eye alignment in detect_and_crop_faces.py stays for human-viewable
crops, where a level, generously padded face is what you actually want to look at.
"""

import os
import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DETECTOR = os.path.join(HERE, "face_detection_yunet_2026may.onnx")
DEFAULT_RECOGNIZER = os.path.join(HERE, "models", "sface.onnx")

# OpenCV's published operating point for SFace. This is a documented property
# of the model, not a knob -- it stays accurate whatever the pipeline defaults to.
COSINE_THRESHOLD = 0.363

# What the CLI and UI actually default to. Deliberately far below the published
# operating point: it accepts nearly every detected face, so treat a match at
# this setting as "worth a look" rather than "verified". Raise it toward 0.363
# for a claim that means something.
DEFAULT_THRESHOLD = 0.1

# Detection confidence floor. 0.5 keeps marginal faces -- small, angled, or
# partly occluded -- that 0.9 discards.
DEFAULT_CONF = 0.5

# Longest side an image may have before detection. This is a memory setting as
# much as a speed one: YuNet's DNN buffers scale with input resolution, and on
# one 2347x1760 probe the detector's own allocations dominated everything else
# in the process.
#
#   cap    resolution   process RSS   detect   confidence
#   3000   1760x2347       457 MB     0.26s      0.955
#   2000   1499x2000       362 MB     0.15s      0.950
#   1600   1199x1599       296 MB     0.11s      0.947
#   1280    959x1280       237 MB     0.10s      0.953
#   1024    767x1024       203 MB     0.07s      0.941
#
# Confidence is flat across that range while memory varies by 250MB, so the
# extra resolution buys nothing -- faces become 112x112 crops regardless. The
# ceiling matters because a 512MB container has roughly 250MB left after the
# models load. Raise MAX_IMAGE_SIDE on a larger host if you need small or
# distant faces, which is the one thing downscaling genuinely costs.
MAX_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1600"))
CANDIDATE_MAX_SIDE = int(os.getenv("MAX_CANDIDATE_SIDE", "1280"))


def downscale(image, max_side=MAX_SIDE):
    """Shrink an image so its longest side is at most max_side. No-op if smaller."""
    if image is None:
        return None
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


class FaceEncoder:
    def __init__(self, detector=DEFAULT_DETECTOR, recognizer=DEFAULT_RECOGNIZER,
                 conf=DEFAULT_CONF, nms=0.3, top_k=5000):
        for p in (detector, recognizer):
            if not os.path.exists(p):
                raise FileNotFoundError(f"model weights not found: {p}")
        self.det = cv2.FaceDetectorYN.create(detector, "", (0, 0), conf, nms, top_k)
        self.rec = cv2.FaceRecognizerSF.create(recognizer, "")

    def detect(self, image):
        """Return raw YuNet rows (15 floats each), highest confidence first."""
        h, w = image.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(image)
        if faces is None:
            return []
        return sorted(faces, key=lambda r: float(r[-1]), reverse=True)

    def embed(self, image, face_row):
        aligned = self.rec.alignCrop(image, face_row)
        return self.rec.feature(aligned).flatten().astype(np.float32)

    def encode_primary(self, image):
        """Encode the single most confident face. Returns (embedding, row) or (None, None).

        The probe image should contain one clear subject; if it holds a crowd we
        deliberately take only the dominant face rather than guessing an identity.
        """
        faces = self.detect(image)
        if not faces:
            return None, None
        row = faces[0]
        return self.embed(image, row), row

    def encode_file(self, path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"could not read image: {path}")
        return self.encode_primary(image)


def preview_crop(image, face_row, pad=0.45, top_extra=0.18):
    """A human-viewable crop of a detected face.

    Deliberately separate from alignCrop(). alignCrop produces the 112x112
    framing SFace was trained on -- tight, chin-to-brow, no context -- which is
    correct for the embedding and unpleasant to look at. Widening *that* would
    change every cosine score, so this pads the raw YuNet box instead and
    leaves the recognition path untouched.

    Extra padding goes above the box because YuNet's box starts around the brow
    line; without it the crop clips the top of the head. The result is clamped
    to the image, so a face near an edge yields an off-centre crop rather than
    a black border.
    """
    h, w = image.shape[:2]
    x, y, bw, bh = (float(v) for v in face_row[0:4])
    dx, dy = bw * pad, bh * pad
    x1 = int(max(0, x - dx))
    y1 = int(max(0, y - dy - bh * top_extra))
    x2 = int(min(w, x + bw + dx))
    y2 = int(min(h, y + bh + dy))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def is_match(a, b, threshold=COSINE_THRESHOLD):
    s = cosine(a, b)
    return s >= threshold, s
