import os
import numpy as np
import json
import csv
import yaml
import cv2
from tqdm import tqdm
from scripts.pixelselection import select_pixels


def resize(img_path, mask_path, size=1024):
    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        raise ValueError(f"Error loading image or mask at {img_path} or {mask_path}")

    r = np.min([size / img.shape[1], size / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(
        mask,
        (int(mask.shape[1] * r), int(mask.shape[0] * r)),
        interpolation=cv2.INTER_NEAREST,
    )
    return img, mask


def setup_prompt_points(
    mask,
    class_id,
    num_pos_points=5,
    num_neg_points=6,
    point_prompt_strat="laplace",  # Options: "fibonacci", "laplace", "shuffle", "gaussian", "centroid", "kmeans", "fps"
    edge_margin=80,
):
    """
    Returns:
        input_points: list of (x, y) tuples, positives first then negatives
        input_labels: list of 1s (for positives) then 0s (for negatives)
    """
    height, width = mask.shape

    # Create a valid region mask to ignore edge pixels.
    valid_region = np.zeros_like(mask, dtype=bool)
    valid_region[
        edge_margin : height - edge_margin, edge_margin : width - edge_margin
    ] = True

    # Prepare a positive mask which only considers valid regions.
    positive_mask = (mask == class_id) & valid_region
    # Note: select_pixels expects an np.ndarray with class IDs, so convert boolean to 0/1.
    positive_mask = positive_mask.astype(np.uint8) * class_id

    # Use the provided strategy to select positive points.
    if np.count_nonzero(positive_mask) > 0:
        try:
            pos_pts = select_pixels(
                positive_mask,
                class_id=class_id,
                num_points=num_pos_points,
                selection_method=point_prompt_strat,
            )
        except ValueError as e:
            pos_pts = []  # fallback if not enough points are available
    else:
        pos_pts = []

    # For negative points, we use a "shuffle" selection method.
    negative_mask = np.full_like(mask, fill_value=-1, dtype=np.int32)
    negative_mask[((mask != class_id) & valid_region)] = 0
    try:
        neg_pts = select_pixels(
            negative_mask,
            class_id=0,
            num_points=num_neg_points,
            selection_method="shuffle",
        )
    except ValueError as e:
        neg_pts = []

    # Convert numpy arrays to list of tuples.
    pos_pts = (
        [tuple(int(val) for val in pt) for pt in pos_pts]
        if isinstance(pos_pts, (list, np.ndarray))
        else []
    )
    neg_pts = (
        [tuple(int(val) for val in pt) for pt in neg_pts]
        if isinstance(neg_pts, (list, np.ndarray))
        else []
    )

    input_points = pos_pts + neg_pts
    input_labels = [1] * len(pos_pts) + [0] * len(neg_pts)

    return input_points, input_labels


def preprocess_split(
    split,
    src_root,
    dst_root,
    class_id,
    point_prompt_strat,
    num_pos,
    num_neg,
    image_size=1024,
):
    # Paths
    image_dir = os.path.join(src_root, split, "images")
    mask_dir = os.path.join(src_root, split, "masks")

    out_img_dir = os.path.join(dst_root, split, "images")
    out_mask_dir = os.path.join(dst_root, split, "masks")
    out_prompt_dir = os.path.join(dst_root, split, "prompts")
    manifest_path = os.path.join(dst_root, split, f"manifest_{split}.csv")

    # Ensure output directories exist
    for d in [out_img_dir, out_mask_dir, out_prompt_dir]:
        os.makedirs(d, exist_ok=True)

    print(
        f"[{split.upper()}] Processing {len(os.listdir(image_dir))} images from: {image_dir}"
    )

    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "mask_path", "input_points", "input_labels"])

        for img_name in tqdm(sorted(os.listdir(image_dir)), desc=f"Processing {split}"):
            img_path = os.path.join(image_dir, img_name)
            base_name, _ = os.path.splitext(img_name)
            mask_path = os.path.join(mask_dir, base_name + ".png")

            try:
                img_resized, ann_mask_resized = resize(img_path, mask_path, image_size)
            except Exception as e:
                print(f"⚠️ Error processing {img_name}: {e}")
                continue

            # Setting up prompt points
            input_pts, input_labels = setup_prompt_points(
                mask=ann_mask_resized,
                class_id=class_id,
                num_pos_points=num_pos,
                num_neg_points=num_neg,
                point_prompt_strat=point_prompt_strat,
                edge_margin=100,
            )

            # Only keep class_id in binary mask as Ground Truth
            binary_mask = (ann_mask_resized == class_id).astype(np.uint8) * 255
            # Output file paths
            out_img_path = os.path.join(out_img_dir, img_name)
            out_mask_path = os.path.join(out_mask_dir, img_name)
            out_prompt_path = os.path.join(out_prompt_dir, base_name + ".json")

            # Save image and mask
            cv2.imwrite(out_img_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_mask_path, binary_mask)

            # Save prompt JSON
            with open(out_prompt_path, "w") as f:
                json.dump({"input_points": input_pts, "input_labels": input_labels}, f)

            # Log to manifest
            writer.writerow(
                [
                    out_img_path,
                    out_mask_path,
                    json.dumps(input_pts),
                    json.dumps(input_labels),
                ]
            )


if __name__ == "__main__":
    with open("sam2/configs/fine_tune_sidewalk/preprocess_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, config["src_root"]))
    DST_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, config["dst_root"]))
    IMAGE_SIZE = config["size"]
    SPLITS = config["splits"]
    CLASS_ID = config["class_id"]
    POINT_PROMPT_STRATEGY = config["point_prompt_strategy"]
    NUM_POS_POINTS = config["num_pos_points"]
    NUM_NEG_POINTS = config["num_neg_points"]

    for folder in SPLITS:
        preprocess_split(
            folder,
            SRC_ROOT,
            DST_ROOT,
            CLASS_ID,
            POINT_PROMPT_STRATEGY,
            NUM_POS_POINTS,
            NUM_NEG_POINTS,
            IMAGE_SIZE,
        )
