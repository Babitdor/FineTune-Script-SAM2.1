import cv2
import numpy as np
import torch
from scripts.pixelselection import select_pixels
import matplotlib.pyplot as plt


def read_batch(data, strategy="laplace", num_pts=4, visualize=False):
    """
    Read random image and its annotation from the dataset (SENSATION)
    Returns:
        image: resized RGB image [H, W, C]
        masks: list of binary masks [N, H, W]
        points: corresponding points [N, 1, 2]
        labels: ones [N, 1]
    """
    # Select random entry
    while True:
        ent = data[np.random.randint(len(data))]
        img_path = ent["image_path"]
        mask_path = ent["mask_path"]

        # Read image and mask
        img = cv2.imread(img_path)[..., ::-1]  # Convert BGR to RGB
        ann_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale

        # Resize image and mask
        r = min(1024 / img.shape[1], 1024 / img.shape[0])  # Scaling factor
        new_size = (int(img.shape[1] * r), int(img.shape[0] * r))
        img = cv2.resize(img, new_size)
        mat_map = cv2.resize(ann_map, new_size, interpolation=cv2.INTER_NEAREST)

        # Get binary masks and points
        inds = np.unique(mat_map)
        inds = inds[inds != 0]

        if len(inds) == 0:
            continue  # Retry another sample if no object masks

        points = []
        masks = []

        for ind in inds:
            mask = (mat_map == ind).astype(np.uint8)
            masks.append(mask)

            # Get points using the specified selection method
            try:
                selected_points = select_pixels(
                    mat_map, ind, num_points=num_pts, selection_method=strategy
                )
                # print(selected_points[0])
                if len(selected_points) > 0:  # type: ignore
                    points.append(selected_points.tolist())  # type: ignore # Format as [[x, y]]
                else:
                    points.append([[0, 0]] * num_pts)  # Fallback if empty selection
            except ValueError:  # In case there aren't enough points
                points.append([[0, 0]] * num_pts)  # Fallback if error
        # Visualization
        if visualize:
            # Create a copy of the image for visualization
            vis_img = img.copy()

            # Overlay masks with random colors
            for i, mask in enumerate(masks):
                color = np.random.randint(0, 255, 3).tolist()
                vis_img[mask == 1] = vis_img[mask == 1] * 0.5 + np.array(color) * 0.5

                # Draw points for this mask
                for point in points[i]:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(vis_img, (x, y), 5, color, -1)
                    cv2.circle(vis_img, (x, y), 5, (255, 255, 255), 1)  # White border

            # Display the visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(vis_img)
            plt.title(f"Image with {len(masks)} objects (points shown in color)")
            plt.axis("off")
            plt.show()

        # print(np.ones([len(masks), num_pts]))
        return (
            img,  # [H, W, 3]
            np.array(masks),  # [N, H, W]
            np.array(points),  # [N, 1, 2]
            np.ones([len(masks), num_pts]),  # [N, 1]
        )
