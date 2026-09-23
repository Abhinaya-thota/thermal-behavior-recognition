import os
import json
import shutil
import argparse
from pathlib import Path

# =========================================================
# CONFIG — override with CLI args, or edit the defaults below
# to point at your local copy of the FLIR ADAS dataset.
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--flir-root", default="data/FLIR_ADAS", help="Root folder of the downloaded FLIR ADAS dataset")
parser.add_argument("--out-root", default="data/flir_yolo", help="Where to write the converted YOLO-format dataset")
args = parser.parse_args()

FLIR_ROOT = args.flir_root

# Thermal image folders
TRAIN_IMG_DIR = os.path.join(FLIR_ROOT, "train", "thermal_8_bit")
VAL_IMG_DIR   = os.path.join(FLIR_ROOT, "val", "thermal_8_bit")

# Annotation files
# Change these if your filenames are different
TRAIN_JSON = os.path.join(FLIR_ROOT, "train", "thermal_annotations.json")
VAL_JSON   = os.path.join(FLIR_ROOT, "val", "thermal_annotations.json")

# YOLO output dataset
OUT_ROOT = args.out_root
# =========================================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def coco_to_yolo_bbox(bbox, img_w, img_h):
    # COCO bbox = [x_min, y_min, width, height]
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm


def convert_split(img_dir, ann_json, out_images, out_labels):
    print(f"\nConverting split from: {img_dir}")
    print(f"Using annotation file: {ann_json}")

    ensure_dir(out_images)
    ensure_dir(out_labels)

    if not os.path.exists(ann_json):
        raise FileNotFoundError(f"Annotation file not found: {ann_json}")

    with open(ann_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    anns_by_image = {}  
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    print("Categories found:", categories)

    copied = 0
    person_images = 0

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img_path = os.path.join(img_dir, os.path.basename(file_name))
        if not os.path.exists(src_img_path):
            continue

        # Copy image
        dst_img_path = os.path.join(out_images, os.path.basename(file_name))
        shutil.copy2(src_img_path, dst_img_path)

        label_lines = []
        has_person = False

        for ann in anns_by_image.get(img_id, []):
            cat_name = categories.get(ann["category_id"], "").lower()

            # Keep only person
            if cat_name != "person":
                continue

            has_person = True

            x_c, y_c, w_n, h_n = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)

            # YOLO class 0 = person
            if w_n > 0 and h_n > 0:
                label_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        if has_person:
            person_images += 1

        label_path = os.path.join(out_labels, Path(dst_img_path).stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))

        copied += 1

    print(f"Images copied: {copied}")
    print(f"Images containing person labels: {person_images}")
    print(f"Labels written to: {out_labels}")


def main():
    train_out_images = os.path.join(OUT_ROOT, "images", "train")
    train_out_labels = os.path.join(OUT_ROOT, "labels", "train")

    val_out_images = os.path.join(OUT_ROOT, "images", "val")
    val_out_labels = os.path.join(OUT_ROOT, "labels", "val")

    convert_split(TRAIN_IMG_DIR, TRAIN_JSON, train_out_images, train_out_labels)
    convert_split(VAL_IMG_DIR, VAL_JSON, val_out_images, val_out_labels)

    print("\nDONE.")
    print("YOLO dataset created at:", OUT_ROOT)


if __name__ == "__main__":
    main()