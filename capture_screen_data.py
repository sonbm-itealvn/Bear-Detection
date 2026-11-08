"""
capture_screen_data.py

Generate REAL screen-capture training data by displaying bear images
fullscreen and capturing them with a camera (matches partner's test setup).

This produces training data in the EXACT domain the model will be deployed in:
camera viewing a monitor displaying trail-cam footage.

Two modes:
  1. Auto mode: display each image, capture via connected camera (OpenCV)
  2. Manual mode: display images one-by-one, you capture externally

Priority: night_vision and distance (small/far bear) images first —
these fail most in camera-capture testing.

Usage:
    # Auto-capture using connected USB camera (device 0)
    python capture_screen_data.py \\
        --source bear_dataset_v_final_final/train/bear \\
        --output screen_captured_bears \\
        --camera 0 \\
        --priority night_vision distance

    # Display only (you capture with external device)
    python capture_screen_data.py \\
        --source bear_dataset_v_final_final/train/bear \\
        --display-only \\
        --delay 3
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path


def get_priority_images(source_dir, priority_keywords, n_total):
    """Select images, prioritizing those matching priority keywords (night/distance)."""
    source_dir = Path(source_dir)
    all_imgs = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        all_imgs.extend(source_dir.rglob(ext))

    if not all_imgs:
        print(f"No images found in {source_dir}")
        return []

    # Score by priority keyword match in filename
    def priority_score(fp):
        name = fp.name.lower()
        return sum(1 for kw in priority_keywords if kw in name)

    # Sort: priority images first, then random within each group
    random.shuffle(all_imgs)
    all_imgs.sort(key=priority_score, reverse=True)

    selected = all_imgs[:n_total] if n_total > 0 else all_imgs
    n_priority = sum(1 for fp in selected if priority_score(fp) > 0)
    print(f"Selected {len(selected)} images ({n_priority} match priority keywords)")
    return selected


def auto_capture(images, output_dir, camera_id=0, display_time=1.5, warmup=0.5):
    """Display each image fullscreen, capture via camera."""
    import cv2
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_id}")
        print("Try: --camera 1, or use --display-only mode")
        return 0

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    win = "DISPLAY"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    captured = 0
    for i, fp in enumerate(images):
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # Display fullscreen
        cv2.imshow(win, img)
        cv2.waitKey(int(warmup * 1000))  # let display settle

        # Flush camera buffer (get fresh frame)
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            print(f"  Capture failed for {fp.name}")
            continue

        # Save captured frame
        dst = output_dir / f"screencap_{fp.stem}_{i:05d}.jpg"
        cv2.imwrite(str(dst), frame)
        captured += 1

        if captured % 50 == 0:
            print(f"  Captured {captured}/{len(images)}")

        # Hold display
        cv2.waitKey(int((display_time - warmup) * 1000))

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {captured} screen-capture images -> {output_dir}")
    return captured


def display_only(images, delay=3.0):
    """Display images one by one for external capture."""
    import cv2

    win = "DISPLAY - capture externally, press any key for next"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for i, fp in enumerate(images):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        cv2.imshow(win, img)
        print(f"  [{i+1}/{len(images)}] {fp.name}")
        key = cv2.waitKey(int(delay * 1000))
        if key == 27:  # ESC to quit
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Source bear images dir")
    parser.add_argument("--output", default="screen_captured_bears")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--n", type=int, default=500, help="Number to capture (0=all)")
    parser.add_argument("--priority", nargs="+",
                        default=["night", "ir", "dark", "distance", "far"],
                        help="Filename keywords to prioritize")
    parser.add_argument("--display-time", type=float, default=1.5,
                        help="Seconds to display each image")
    parser.add_argument("--display-only", action="store_true",
                        help="Only display, capture externally")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Display-only mode: seconds per image")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SCREEN CAPTURE DATA GENERATION")
    print(f"  Source: {args.source}")
    print(f"  Priority: {args.priority}")
    print(f"{'='*60}\n")

    images = get_priority_images(args.source, args.priority, args.n)
    if not images:
        return

    if args.display_only:
        print("Display-only mode. Capture each image with your external camera.")
        print("Press any key for next image, ESC to quit.\n")
        display_only(images, delay=args.delay)
    else:
        auto_capture(images, args.output, camera_id=args.camera,
                     display_time=args.display_time)

    print(f"\nNext: verify captured images, then merge into dataset as bear class")


if __name__ == "__main__":
    main()
