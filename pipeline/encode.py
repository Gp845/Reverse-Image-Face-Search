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
# Smallest face worth encoding. Detection confidence is not a usable proxy for
# embedding quality: shrinking one face until it was 24px wide kept YuNet's
# confidence at 0.911 while its embedding drifted to 0.80 against the same face
# at full size (0.91 at 48px, 0.94 at 64px). Below this, a "match" says more
# about resolution than identity.
MIN_FACE_PX = 48

# Two detections of the same person in one image produce near-identical crops
# and score far above this; two different people in a group photo score well
# below it. Used only to collapse duplicates, never to claim a match.
SAME_FACE_COSINE = 0.6

MAX_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "1600"))

# Detection resolution when several faces are wanted. Faces in a group photo
# are small, and downscaling loses them outright rather than merely blurring
# them: on a 1760x2347 photo the detector found one face at 1280px and two at
# 2000px. Memory is the reason this is not simply the default -- the same sweep
# measured 257MB at 1280px against 389MB at 2000px, and a 512MB container has
# to fit the match-back too. Lower MAX_GROUP_SIDE if a deployment is tight.
GROUP_MAX_SIDE = int(os.getenv("MAX_GROUP_SIDE", "2000"))
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

    def encode_rows(self, image, rows, max_faces=None, min_size=MIN_FACE_PX,
                    same_face=SAME_FACE_COSINE):
        """Encode every distinct face in the image, largest first.

        Ordered by face area rather than detector confidence, because
        confidence stays high on faces far too small to encode reliably, while
        area tracks embedding quality closely. Faces narrower than min_size are
        dropped for the same reason.

        Duplicates are collapsed by comparing embeddings: a person detected
        twice -- in a reflection, a collage, or by two overlapping boxes NMS
        did not merge -- would otherwise consume a search slot twice over.

        Takes rows already detected, so detection can run on a cheap downscaled
        copy while the embeddings are cut from full-resolution pixels.

        Returns [(embedding, row), ...].
        """
        rows = sorted(rows, key=lambda r: float(r[2]) * float(r[3]), reverse=True)
        kept = []
        for row in rows:
            if float(row[2]) < min_size or float(row[3]) < min_size:
                continue
            emb = self.embed(image, row)
            if any(cosine(emb, prev) >= same_face for prev, _ in kept):
                continue
            kept.append((emb, row))
            if max_faces and len(kept) >= max_faces:
                break
        return kept

    def encode_all(self, image, **kw):
        """Detect and encode every distinct face in one image."""
        return self.encode_rows(image, self.detect(image), **kw)

    def encode_file(self, path):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"could not read image: {path}")
        return self.encode_primary(image)


def scale_row(row, factor):
    """Rescale a YuNet row to a different resolution of the same image.

    The first 14 values are the box and the five landmarks, all in pixels; the
    last is the confidence score and must not be touched. Used to detect on a
    downscaled copy -- which is what keeps memory bounded -- and then encode
    from the original pixels, so a face that is 48px in the small copy is cut
    from the 70px it actually occupies in the source.
    """
    out = row.copy()
    out[0:14] = row[0:14] * factor
    return out


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
