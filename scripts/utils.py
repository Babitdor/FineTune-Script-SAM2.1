import os
import numpy as np
import random
import cv2
import torch


# To ensure if the path exist
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Function to Resize the mask and Image
def resize(img_path, mask_path, size):
    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img_resized = cv2.resize(img, size)
    mask_resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return img_resized, mask_resized


# Save the binary mask of sidewalk
def save_binary_masks_of_sidewalk(mask, out_dir, base_name):

    sidewalk_class_id = 2
    image_dir = os.path.join(out_dir, base_name)
    ensure_dir(image_dir)

    mask_path = os.path.join(image_dir, f"sidewalk_mask_{base_name}.png")
    if sidewalk_class_id in np.unique(mask):
        binary_mask = (mask == sidewalk_class_id).astype(np.uint8) * 255
    else:
        # An empty mask is created if sidewalk class is not present
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.imwrite(mask_path, binary_mask)
    return mask_path


# Function to pick point prompts
def setup_prompt_points(mask, strategy="random", num_pos_points=5, num_neg_points=6):
    """
    Returns:
        input_points: list of (x, y) tuples, positives first then negatives
        input_labels: list of 1s (for positives) then 0s (for negatives)
    """
    sidewalk_class_id = 2
    pos_pts = np.argwhere(mask == sidewalk_class_id)
    neg_pts = np.argwhere(mask != sidewalk_class_id)

    def pick(points, strategy):
        if len(points) == 0:
            return None
        if strategy == "center":
            idx = len(points) // 2
        elif strategy == "random":
            idx = random.randint(0, len(points) - 1)
        elif strategy == "hybrid":
            if len(points) > 1:
                if random.choice([True, False]):
                    idx = len(points) // 2
                else:
                    idx = random.randint(0, len(points) - 1)
            else:
                idx = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        x = int(points[idx][1])
        y = int(points[idx][0])
        return (x, y)

    # Collect positive points and labels
    pos_points = []
    pos_labels = []
    for _ in range(num_pos_points):
        pos_point = pick(pos_pts, strategy)
        if pos_point and pos_point not in pos_points:
            pos_points.append(pos_point)
            pos_labels.append(1)

    # Collect negative points and labels
    neg_points = []
    neg_labels = []
    for _ in range(num_neg_points):
        neg_point = pick(neg_pts, "random")
        if neg_point and neg_point not in neg_points:
            neg_points.append(neg_point)
            neg_labels.append(0)

    # Concatenate positives and negatives
    input_points = pos_points + neg_points
    input_labels = pos_labels + neg_labels

    return input_points, input_labels


def read_data(batch):

    image = batch["image"][0]  # shape expected by SAM2.1: [C, H, W] or [H, W, C]
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    mask = batch["mask"][0]
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    input_points = batch["input_points"][0]
    input_labels = batch["input_labels"][0]
    return image, mask, input_points, input_labels


def post_process(mask, thresh=0.5, open_k=3, close_k=5, smooth=False, iterations=1):

    mask = (mask > thresh).astype(np.uint8)

    if smooth:
        mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        mask = (mask > thresh).astype(np.uint8)

    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)

    for _ in range(iterations):
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)
        )
        return mask
