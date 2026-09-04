#!/usr/bin/env python3
"""
Detect all faces in an image using OpenCV's DNN-based face detector (YuNet),
and extract them using a strict 2-Point (Eye) Alignment. 

This guarantees the eyes are perfectly horizontal and avoids the artificial 
tilts caused by 5-point Procrustes alignment on non-frontal faces.
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Model file (must be alongside this script, or pass --model)
DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "face_detection_yunet_2026may.onnx"
)


def load_face_detector(model: str, conf_threshold: float, nms_threshold: float, top_k: int):
    if not os.path.exists(model):
        sys.exit(f"Model weights not found: {model}")
    detector = cv2.FaceDetectorYN.create(
        model, "", (0, 0),
        score_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        top_k=top_k,
    )
    return detector


def detect_faces(detector, image: np.ndarray):
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)

    boxes = []
    if faces is None:
        return boxes

    for face in faces:
        x, y, bw, bh = face[0:4]
        conf = face[-1]
        
        # Extract eyes (Indexes 4,5 and 6,7 in YuNet)
        eye1 = (face[4], face[5])
        eye2 = (face[6], face[7])

        if bw > 0 and bh > 0:
            boxes.append({
                "box": (x, y, bw, bh),
                "eye1": eye1,
                "eye2": eye2,
                "conf": float(conf)
            })
            
    return boxes


def pad_box(x1, y1, x2, y2, w, h, pad_frac):
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def main():
    parser = argparse.ArgumentParser(description="Detect, align (deskew), and crop faces.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("outdir", nargs="?", default="cropped_faces", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold (0-1)")
    parser.add_argument("--nms", type=float, default=0.3, help="NMS IoU threshold (0-1)")
    parser.add_argument("--top-k", type=int, default=5000, help="Max candidate boxes before NMS")
    parser.add_argument("--pad", type=float, default=0.2, help="Padding fraction around each face")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-align", action="store_true", help="Disable facial alignment (tilt fixing)")
    parser.add_argument("--no-boxes-image", action="store_true", help="Skip saving annotated full image")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Could not read image: {args.image}")

    detector = load_face_detector(args.model, args.conf, args.nms, args.top_k)
    faces_data = detect_faces(detector, image)

    os.makedirs(args.outdir, exist_ok=True)
    h, w = image.shape[:2]

    print(f"Found {len(faces_data)} face(s) in {args.image}")

    annotated = image.copy()
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    for idx, face in enumerate(faces_data, start=1):
        x, y, bw, bh = face["box"]
        conf = face["conf"]
        ex1, ey1 = face["eye1"]
        ex2, ey2 = face["eye2"]

        if not args.no_align:
            # 1. Enforce left-to-right eye order
            if ex1 > ex2:
                ex1, ey1, ex2, ey2 = ex2, ey2, ex1, ey1
            
            dx = ex2 - ex1
            dy = ey2 - ey1
            
            # cv2.getRotationMatrix2D takes a POSITIVE angle for a counter-clockwise
            # rotation as displayed; it already accounts for the y-down image axis,
            # so the raw eye-line angle is passed through unnegated. Negating it
            # rotates the wrong way and doubles the tilt instead of removing it.
            angle = np.degrees(np.arctan2(dy, dx))
            
            # 2. Get rotation matrix anchored perfectly to the center of the eyes
            eye_cx = (ex1 + ex2) / 2.0
            eye_cy = (ey1 + ey2) / 2.0
            M = cv2.getRotationMatrix2D((eye_cx, eye_cy), angle, 1.0)
            
            # 3. Find where the original bounding box center ends up after rotation
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            new_cx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
            new_cy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
            
            # 4. Calculate final crop dimensions based on padded bounding box
            crop_w = int(bw * (1.0 + 2.0 * args.pad))
            crop_h = int(bh * (1.0 + 2.0 * args.pad))
            
            # 5. Shift the matrix so the face center is perfectly in the middle of our crop
            M[0, 2] += (crop_w / 2.0) - new_cx
            M[1, 2] += (crop_h / 2.0) - new_cy
            
            # 6. Apply Affine transform (Rotation + Translation + Cropping all at once)
            crop = cv2.warpAffine(
                image, M, (crop_w, crop_h), 
                flags=cv2.INTER_CUBIC, 
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
        else:
            # Unaligned cropping fallback
            x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
            px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, w, h, args.pad)
            crop = image[py1:py2, px1:px2]

        out_path = os.path.join(args.outdir, f"{base_name}_face{idx}.jpg")
        cv2.imwrite(out_path, crop)
        
        tilt_text = f", tilt={angle:.1f}°" if not args.no_align else ""
        print(f"  face {idx}: conf={conf:.2f}{tilt_text} -> {out_path}")

        # Visual Debugging on the Annotated Image
        x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{idx} ({conf:.2f})", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw eyes and the alignment line
        cv2.circle(annotated, (int(ex1), int(ey1)), 2, (255, 0, 0), -1)
        cv2.circle(annotated, (int(ex2), int(ey2)), 2, (0, 0, 255), -1)
        cv2.line(annotated, (int(ex1), int(ey1)), (int(ex2), int(ey2)), (0, 255, 255), 1)

    if not args.no_boxes_image:
        annotated_path = os.path.join(args.outdir, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)
        print(f"Annotated image saved to: {annotated_path}")

if __name__ == "__main__":
    main()