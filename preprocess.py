import os
import numpy as np
import json
import csv
import yaml
import cv2
from tqdm import tqdm
from scripts.utils import (
    ensure_dir,
    resize,
    save_binary_masks_of_sidewalk,
    setup_prompt_points,
)


def preprocess_split(
    split, src_root, dst_root, num_pos, num_neg, point_strat, size=(1024, 1024)
):
    image_dir = os.path.join(src_root, split, "images")
    mask_dir = os.path.join(src_root, split, "masks")

    out_img_dir = os.path.join(dst_root, split, "images")
    out_mask_dir = os.path.join(dst_root, split, "masks")
    out_prompt_dir = os.path.join(dst_root, split, "prompts")

    manifest_path = os.path.join(dst_root, split, f"manifest_{split}.csv")

    print(image_dir)
    ensure_dir(out_img_dir)
    ensure_dir(out_mask_dir)
    ensure_dir(out_prompt_dir)

    image_files = sorted(os.listdir(image_dir))

    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "mask_path", "input_points", "input_labels"])

        for img_name in tqdm(image_files, desc=f"Processing {split}"):

            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name).replace(
                ".jpg", ".png"
            )  # assuming same name

            img_resized, mask_binary = resize(img_path, mask_path, size)

            input_pts, input_labels = setup_prompt_points(
                mask_binary, point_strat, num_pos, num_neg
            )

            # Save processed image
            out_img_path = os.path.join(out_img_dir, img_name)
            out_mask_path = os.path.join(out_mask_dir, img_name)
            out_prompt_path = os.path.join(
                out_prompt_dir,
                img_name.replace(".jpg", ".json").replace(".png", ".json"),
            )

            cv2.imwrite(out_img_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_mask_path, mask_binary.astype(np.uint8))

            # Save binary masks for sidewalk class
            base_name = os.path.splitext(img_name)[0]
            side_walk_mask_path = save_binary_masks_of_sidewalk(
                mask_binary, out_mask_dir, base_name
            )

            with open(out_prompt_path, "w") as f:
                json.dump({"input_points": input_pts, "input_labels": input_labels}, f)

            writer.writerow(
                [
                    out_img_path,
                    side_walk_mask_path,
                    json.dumps(input_pts),
                    json.dumps(input_labels),
                ]
            )


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, config["src_root"]))
    DST_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, config["dst_root"]))
    SIZE = tuple(config["size"])
    SPLITS = config["splits"]
    SIDEWALK_CLASS_ID = config["sidewalk_class_id"]
    NUM_POS_POINTS = config["num_pos_points"]
    NUM_NEG_POINTS = config["num_neg_points"]
    PROMPT_STRAT = config["point_propmpt_strategy"]

for folder in SPLITS:
    preprocess_split(
        folder,
        SRC_ROOT,
        DST_ROOT,
        NUM_POS_POINTS,
        NUM_NEG_POINTS,
        PROMPT_STRAT,
        SIZE,
    )
