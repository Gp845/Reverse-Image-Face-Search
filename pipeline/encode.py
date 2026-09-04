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

# OpenCV's published operating point for SFace.
COSINE_THRESHOLD = 0.363


class FaceEncoder:
    def __init__(self, detector=DEFAULT_DETECTOR, recognizer=DEFAULT_RECOGNIZER,
                 conf=0.9, nms=0.3, top_k=5000):
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


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def is_match(a, b, threshold=COSINE_THRESHOLD):
    s = cosine(a, b)
    return s >= threshold, s
