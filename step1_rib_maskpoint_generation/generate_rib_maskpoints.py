"""Step 1 — generate rib maskpoint JSONs for every image in a folder.

Segments the rib cage with a pretrained EfficientUNet, then samples
well-spread points inside the rib mask (farthest-point sampling). Each image
produces <output>/<stem>.json:
    {"image_size": [w, h], "points_list": [[x, y], ...]}

Usage:
    python generate_rib_maskpoints.py --input <image_dir> \
        [--output <json_dir>] [--device cuda:0] [--sample-points 128]
"""
import argparse
import glob
import json
import os

import torch
from PIL import Image
from torchvision import transforms

from segmentation.efficient_unet import EfficientUNet, inference_with_resize_and_restore
from segmentation.point_sampling import sample_points_in_mask_batch

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "weight.pt")
SUPPORTED_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def list_images(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.*"))
    return sorted(p for p in all_files if os.path.splitext(p)[1].lower() in SUPPORTED_EXTS)


def main(args):
    output_dir = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)), "maskpoint")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input:  {args.input}")
    print(f"Output: {output_dir}")

    print(f"Loading rib segmentation weights from {args.weights} ...")
    rib_model = EfficientUNet(device=args.device)
    rib_model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    rib_model.eval()
    rib_model.to(args.device)

    transform = transforms.ToTensor()
    image_paths = list_images(args.input)
    print(f"Found {len(image_paths)} images")

    for index, img_path in enumerate(image_paths, 1):
        img_name = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_name)
        out_json_path = os.path.join(output_dir, f"{base_name}.json")

        try:
            img_pil = Image.open(img_path).convert("RGB")
            w, h = img_pil.size
            img_tensor = transform(img_pil).unsqueeze(0).to(args.device)

            rib_mask = inference_with_resize_and_restore(rib_model, img_tensor)
            points_list = sample_points_in_mask_batch(
                rib_mask, sample_points=args.sample_points,
                min_edge_dist=args.min_edge_dist,
            )
            points = points_list[0].tolist() if len(points_list) > 0 else []

            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump({"image_size": [w, h], "points_list": points}, f, indent=4)
            print(f"[{index}/{len(image_paths)}] {img_name} -> {out_json_path}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Folder of input CXR images")
    parser.add_argument("--output", default=None,
                        help="Output JSON folder (default: <input parent>/maskpoint)")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Rib segmentation model weights")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-points", type=int, default=128,
                        help="Rib points sampled per image")
    parser.add_argument("--min-edge-dist", type=float, default=6.0,
                        help="Min distance (px) a point keeps from the rib-mask boundary")
    args = parser.parse_args()
    with torch.no_grad():
        main(args)
