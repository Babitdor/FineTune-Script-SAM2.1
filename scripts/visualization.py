import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml


def visualize_points_on_mask(mask_path, prompts):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[WARNING] Could not read mask: {mask_path}")
        return

    # Convert mask to color for visualization
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for prompt in prompts:
        x, y = prompt["coord"]
        if prompt["label"] == 1:
            color = (0, 255, 0)  # Green for positive
        else:
            color = (255, 0, 0)  # Red for negative
        cv2.circle(mask_color, (x, y), radius=6, color=color, thickness=-1)

    # Convert BGR (OpenCV) to RGB for matplotlib
    mask_color_rgb = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask_color_rgb)
    plt.axis("off")
    plt.title(os.path.basename(mask_path))
    plt.show()


if __name__ == "__main__":

    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    GT_MASK = config["gt_mask"]
    PROMPTS_PATH = config["prompts_path"]

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    masks_root = os.path.join(
        SCRIPT_DIR,
        GT_MASK,
    )

    prompts_root = os.path.join(SCRIPT_DIR, PROMPTS_PATH)

    subfolders = [
        f for f in os.listdir(masks_root) if os.path.isdir(os.path.join(masks_root, f))
    ]

    if not subfolders:
        print("No mask subfolders found.")
        exit()

    # Pick 5 random folders
    chosen_folders = random.sample(subfolders, min(5, len(subfolders)))

    for folder in chosen_folders:
        mask_dir = os.path.join(masks_root, folder)
        prompt_file = os.path.join(prompts_root, f"{folder}.json")

        if not os.path.exists(prompt_file):
            print(f"[WARNING] Missing prompt file: {prompt_file}")
            continue

        with open(prompt_file, "r") as f:
            prompts = json.load(f)

        # Get all mask image files
        mask_files = [
            f
            for f in os.listdir(mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not mask_files:
            print(f"[WARNING] No mask files in {mask_dir}")
            continue

        # Pick 1 random mask per folder to show
        chosen_mask = random.choice(mask_files)
        mask_path = os.path.join(mask_dir, chosen_mask)

        print(f"[INFO] Showing: {mask_path}")
        visualize_points_on_mask(mask_path, prompts)
