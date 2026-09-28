#!/usr/bin/env python3
"""
Vinyasa - Part 1: Object Detection + Masked Crop Generator

Detects furniture in a room photo using YOLOv8-seg, produces a masked
cutout of each object on a neutral gray background, and writes an
object_manifest.json describing what was found (for Part 2 to consume).

Usage:
    python detect_and_crop.py --input room.jpg --output data/detections
    python detect_and_crop.py --input photos_dir/ --output data/detections

Install:
    pip install ultralytics opencv-python numpy
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent  # vinyasa/
DEFAULT_INPUT = ROOT_DIR / "data" / "input"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "detections"
_local_weights = Path(__file__).resolve().parent / "yolov8n-seg.pt"
DEFAULT_WEIGHTS = str(_local_weights) if _local_weights.exists() else "yolov8n-seg.pt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# COCO class names we care about for room/furniture scenes.
# (YOLOv8-seg is trained on COCO-80; these are the relevant subset.)
FURNITURE_CLASSES = {
    "chair": "chair",
    "couch": "sofa",
    "bed": "bed",
    "dining table": "table",
    "tv": "tv",
    "potted plant": "plant",
    "sink": "sink",
    "refrigerator": "refrigerator",
    "toilet": "toilet",
    "book": "book",
    "vase": "vase",
    "clock": "clock",
}

GRAY_BG = (128, 128, 128)  # neutral gray, matches TripoSR expected input style
CONFIDENCE_THRESHOLD = 0.35  # default threshold for most classes

# Some classes (like dining table, when it's really a low coffee table)
# consistently score lower confidence on COCO-pretrained weights even when
# the detection is correct. Give those a lower bar individually rather than
# lowering the threshold for everything.
CLASS_CONFIDENCE_OVERRIDES = {
    "table": 0.25,
}


def get_threshold(category: str) -> float:
    return CLASS_CONFIDENCE_OVERRIDES.get(category, CONFIDENCE_THRESHOLD)


def load_model(weights: str = "yolov8n-seg.pt") -> YOLO:
    """Loads a pretrained YOLOv8 segmentation model (auto-downloads weights)."""
    print(f"Loading model: {weights}")
    return YOLO(weights)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Cleans a raw binary segmentation mask:
    1. Keeps only the largest connected component (drops disconnected
       floating scraps the model included by mistake).
    2. Applies morphological closing to smooth jagged edges and fill
       small interior holes.
    """
    mask_u8 = (mask.astype(np.uint8)) * 255

    # Step 1: largest connected component only.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels > 1:
        # label 0 is background; find the largest non-background component.
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_u8 = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Step 2: morphological closing to smooth edges / fill small holes.
    kernel = np.ones((7, 7), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    return mask_u8 > 127


def make_masked_crop(image: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Given the full image, a per-pixel binary mask (same H,W as image),
    and a bbox (x1,y1,x2,y2), returns a cropped image where only the
    masked object is visible, composited onto a solid gray background.
    """
    x1, y1, x2, y2 = bbox
    mask = clean_mask(mask)

    # Build a 3-channel gray canvas the size of the full image, then
    # copy in only the masked pixels — this keeps mask edges clean.
    gray_canvas = np.full_like(image, GRAY_BG, dtype=np.uint8)
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(bool)
    composited = np.where(mask_3ch, image, gray_canvas)

    # Crop down to the bounding box (with a small padding margin).
    pad = 4
    h, w = image.shape[:2]
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    return composited[cy1:cy2, cx1:cx2]


def process_image(model: YOLO, image_path: Path, crops_dir: Path, start_id: int, debug: bool = False) -> tuple[list[dict], int]:
    """Runs detection on one image, writes crops, returns manifest entries."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  WARNING: could not read {image_path}, skipping")
        return [], start_id

    h, w = image.shape[:2]
    results = model(image, verbose=False)[0]

    entries = []
    obj_id = start_id

    if results.masks is None:
        print(f"  No objects detected in {image_path.name}")
        return [], start_id

    names = results.names
    boxes = results.boxes
    masks = results.masks.data.cpu().numpy()  # (N, mask_h, mask_w)

    if debug:
        print(f"  --- RAW detections for {image_path.name} (before filtering) ---")
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            cls_name = names[cls_id]
            conf = float(boxes.conf[i])
            in_classes = cls_name in FURNITURE_CLASSES
            threshold = get_threshold(FURNITURE_CLASSES[cls_name]) if in_classes else CONFIDENCE_THRESHOLD
            kept = in_classes and conf >= threshold
            print(f"    [{'KEPT' if kept else 'DROPPED'}] {cls_name}  conf={conf:.3f}  threshold={threshold:.2f}"
                  f"{'' if in_classes else '  (not in FURNITURE_CLASSES)'}")
        print("  --- end raw detections ---")

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        cls_name = names[cls_id]
        conf = float(boxes.conf[i])

        if cls_name not in FURNITURE_CLASSES:
            continue
        category = FURNITURE_CLASSES[cls_name]
        if conf < get_threshold(category):
            continue

        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

        # Resize mask to full image resolution (YOLO masks come out at model res).
        mask_resized = cv2.resize(masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bin = mask_resized > 0.5

        crop = make_masked_crop(image, mask_bin, (x1, y1, x2, y2))

        crop_filename = f"obj_{obj_id:03d}_{category}.png"
        cv2.imwrite(str(crops_dir / crop_filename), crop)

        entries.append({
            "id": f"obj_{obj_id:03d}",
            "category": category,
            "confidence": round(conf, 3),
            "crop_file": crop_filename,
            "source_image": image_path.name,
            "bbox": {
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2),
            },
            "source_image_size": {"width": w, "height": h},
        })
        obj_id += 1

    print(f"  {image_path.name}: {len(entries)} object(s) kept "
          f"(of {len(boxes)} raw detections)")
    return entries, obj_id


def main():
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Room photo file or directory of photos")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output dir for crops/ + manifest")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLOv8-seg weights to use")
    parser.add_argument("--debug", action="store_true", help="Print every raw detection before filtering")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    crops_dir = output_dir / "crops"

    if not input_path.exists():
        print(f"Input not found: {input_path}\nPut your room photos in {DEFAULT_INPUT}")
        return

    crops_dir.mkdir(parents=True, exist_ok=True)

    image_paths = (
        sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if input_path.is_dir() else [input_path]
    )
    if not image_paths:
        print(f"No images found at {input_path}")
        return

    model = load_model(args.weights)

    t0 = time.time()
    all_entries = []
    next_id = 1
    for img_path in image_paths:
        entries, next_id = process_image(model, img_path, crops_dir, next_id, debug=args.debug)
        all_entries.extend(entries)
    elapsed = time.time() - t0

    manifest = {
        "generated_by": "detect_and_crop.py",
        "object_count": len(all_entries),
        "objects": all_entries,
    }
    manifest_path = output_dir / "object_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone in {elapsed:.2f}s ({elapsed / len(image_paths):.2f}s/image avg)")
    print(f"Detected {len(all_entries)} object(s) total")
    print(f"Manifest written to: {manifest_path}")
    print(f"Crops written to: {crops_dir}")


if __name__ == "__main__":
    main()