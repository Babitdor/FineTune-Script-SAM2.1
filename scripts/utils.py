import os
import numpy as np
import random
import cv2


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

    Sets up:
        1. Select Positive points based on strategy for our intended class_label binary mask (here in this case (sidewalk) which is class_id = 2)
        2. Select Negative points (1 point) for all other class labels

    """

    sidewalk_class_id = 2
    prompts = []
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
                # Randomly choose between center and random selection
                if random.choice([True, False]):
                    idx = len(points) // 2
                else:
                    idx = random.randint(0, len(points) - 1)
            else:
                idx = 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Convert numpy.int64 to native Python int and return (x, y)
        x = int(points[idx][1])
        y = int(points[idx][0])
        return (x, y)

    # Add positive points
    for _ in range(num_pos_points):
        pos_point = pick(pos_pts, strategy)
        if pos_point and pos_point not in [
            p["coord"] for p in prompts
        ]:  # Avoid duplicates
            prompts.append({"coord": pos_point, "label": 1})

    # Add negative points
    for _ in range(num_neg_points):
        neg_point = pick(neg_pts, "random")  # Always use random for background
        if neg_point and neg_point not in [p["coord"] for p in prompts]:
            prompts.append({"coord": neg_point, "label": 0})

    return prompts
