import os
import numpy as np
import random
import cv2
import torch


def read_data(batch):

    image = batch["image"][0]  # shape expected by SAM2.1: [C, H, W] or [H, W, C]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    mask = batch["mask"][0]
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
