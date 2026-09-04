#!/usr/bin/env python3
"""
Dynamically enhance faces using GFPGAN, with an aggressive pre-scaling 
engine designed specifically to force extreme low-resolution images into 
a format the AI can actually process.
"""

import argparse
import os
import sys
import cv2
import numpy as np

# --- FIX FOR BASICSR / TORCHVISION MISMATCH ---
import torchvision.transforms.functional as TF
sys.modules['torchvision.transforms.functional_tensor'] = TF
# ----------------------------------------------

from gfpgan import GFPGANer


def calculate_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def pre_scale_for_ai(image, target_min_size=256):
    """
    TRICKING THE AI: If the image is too small, the GAN's internal face 
    detector will fail. We use Lanczos-4 interpolation to stretch the 
    raw pixels up to a size the GAN was trained to understand.
    """
    h, w = image.shape[:2]
    min_dim = min(h, w)
    
    if min_dim < target_min_size:
        # Calculate how much we need to scale to reach the target size
        scale_factor = target_min_size / min_dim
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        print(f"  -> PRE-SCALING TRIGGERED: Stretching {w}x{h} to {new_w}x{new_h} so the AI can see it.")
        
        # INTER_LANCZOS4 is mathematically the best traditional algorithm for upscaling
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    return image


def determine_enhancement_weight(image):
    h, w = image.shape[:2]
    max_dim = max(h, w)
    sharpness = calculate_image_quality(image)
    
    print(f"  -> Image Analysis: Size={w}x{h}, Sharpness Score={sharpness:.1f}")

    if max_dim < 256 or sharpness < 75:
        print("  -> Classification: EXTREME DEGRADATION (Applying 100% AI Reconstruction)")
        return 1.0
    elif max_dim > 500 and sharpness > 350:
        print("  -> Classification: HIGH QUALITY (Applying 50% AI Blend)")
        return 0.5
    else:
        weight = 1.0 - (sharpness / 1000.0)
        weight = max(0.6, min(weight, 0.9))
        print(f"  -> Classification: MODERATE DEGRADATION (Applying {weight*100:.0f}% AI Blend)")
        return weight


def main():
    parser = argparse.ArgumentParser(description="Enhance extreme low-res faces using GFPGAN.")
    parser.add_argument("image", help="Path to the input face image")
    parser.add_argument("output", nargs="?", help="Path to save the enhanced image")
    parser.add_argument("--scale", type=int, default=2, help="Final AI upsampling scale factor")
    args = parser.parse_args()

    if not args.output:
        base, ext = os.path.splitext(args.image)
        args.output = f"{base}_extreme_enhance{ext}"

    print("Loading GFPGAN v1.3 model...")
    try:
        restorer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.scale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None 
        )
    except Exception as e:
        sys.exit(f"Failed to load GFPGAN: {e}")

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Error: Could not read {args.image}")

    print(f"\nProcessing {args.image}...")

    # 1. ANALYZE FIRST, on the ORIGINAL pixels. Lanczos upscaling smooths the
    # image and pushes Laplacian variance down, so measuring after the stretch
    # makes every small input look severely degraded and forces weight=1.0
    # (maximum hallucination) as an artifact of our own pre-scaling.
    dynamic_weight = determine_enhancement_weight(img)

    # 2. PRE-PROCESS: stretch tiny images before they ever touch the AI
    img_ready = pre_scale_for_ai(img)

    # 3. ENHANCE
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        img_ready, 
        has_aligned=False, 
        only_center_face=False, 
        paste_back=True,
        weight=dynamic_weight 
    )

    if restored_img is not None:
        cv2.imwrite(args.output, restored_img)
        print(f"\nSuccess! Enhanced image saved to: {args.output}")
    else:
        print("\nThe image was still too degraded for the AI face detector to find landmarks.")

if __name__ == "__main__":
    main()