import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml


def visualize_points_on_mask(mask_path, input_points, input_labels):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[WARNING] Could not read mask: {mask_path}")
        return

    # Convert mask to color for visualization
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for coord, label in zip(input_points, input_labels):
        x, y = coord
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(mask_color, (int(x), int(y)), radius=6, color=color, thickness=-1)

    # Convert BGR (OpenCV) to RGB for matplotlib
    mask_color_rgb = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask_color_rgb)
    plt.axis("off")
    plt.title(os.path.basename(mask_path))
    plt.show()


if __name__ == "__main__":

    with open("./configs/preprocess_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    GT_MASK = config["gt_mask"]
    PROMPTS_PATH = config["prompts_path"]

    masks_root = os.path.normpath(GT_MASK)
    prompts_root = os.path.normpath(PROMPTS_PATH)

    # List all mask files in the flat masks_root
    mask_files = [
        f
        for f in os.listdir(masks_root)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not mask_files:
        print(f"[WARNING] No mask files found in {masks_root}")
        exit()

    # Pick 5 random mask files
    chosen_masks = random.sample(mask_files, min(5, len(mask_files)))

    for mask_file in chosen_masks:
        mask_path = os.path.join(masks_root, mask_file)
        base_name, _ = os.path.splitext(mask_file)
        prompt_file = os.path.join(prompts_root, base_name + ".json")

        if not os.path.exists(prompt_file):
            print(f"[WARNING] Missing prompt file: {prompt_file}")
            continue

        with open(prompt_file, "r") as f:
            prompt_data = json.load(f)
            input_points = prompt_data.get("input_points", [])
            input_labels = prompt_data.get("input_labels", [])

        print(f"[INFO] Showing: {mask_path}")
        visualize_points_on_mask(mask_path, input_points, input_labels)
