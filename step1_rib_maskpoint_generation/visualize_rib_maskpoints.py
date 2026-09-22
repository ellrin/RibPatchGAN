"""Overlay generated rib maskpoints on the source images for manual review.

Usage:
    python visualize_rib_maskpoints.py --input <image_dir> \
        [--maskpoint-dir <json_dir>] [--output <vis_dir>]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SUPPORTED_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def list_images(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.*"))
    return sorted(p for p in all_files if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS)


def main(args):
    parent_dir = os.path.dirname(os.path.abspath(args.input))
    json_dir = args.maskpoint_dir or os.path.join(parent_dir, "maskpoint")
    vis_dir = args.output or os.path.join(parent_dir, "maskpoint_vis")
    os.makedirs(vis_dir, exist_ok=True)

    image_paths = list_images(args.input)
    print(f"Found {len(image_paths)} images to visualize.")

    for index, img_path in enumerate(image_paths, 1):
        img_name = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_name)
        json_path = os.path.join(json_dir, f"{base_name}.json")
        out_vis_path = os.path.join(vis_dir, f"{base_name}_vis.jpg")

        if not os.path.exists(json_path):
            print(f"Warning: JSON not found for {img_name}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                points = json.load(f).get("points_list", [])

            img = Image.open(img_path).convert("RGB")
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            if points:
                plt.scatter([p[0] for p in points], [p[1] for p in points], color="lime", s=10)
            plt.axis("off")
            plt.savefig(out_vis_path, bbox_inches="tight", pad_inches=0, dpi=150)
            plt.close()
            print(f"[{index}/{len(image_paths)}] {img_name} -> {out_vis_path}")
        except Exception as e:
            print(f"Error visualizing {img_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder of input CXR images")
    parser.add_argument("--maskpoint-dir", default=None,
                        help="Folder of maskpoint JSONs (default: <input parent>/maskpoint)")
    parser.add_argument("--output", default=None,
                        help="Output folder (default: <input parent>/maskpoint_vis)")
    main(parser.parse_args())
