import os
import cv2
import json
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch


class SENSATIONDataset(Dataset):
    def __init__(self, manifest_path, root_dir="", transform=None, max_points=10):
        self.data = pd.read_csv(manifest_path)
        self.root_dir = root_dir
        self.transform = transform
        self.max_points = max_points

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        return np.array(mask) // 255

    def _parse_points_and_labels(self, points_str, labels_str):
        # Parse points and labels from JSON strings
        try:
            if pd.isna(points_str) or points_str.strip() == "":
                points = np.zeros((0, 2), dtype=np.float32)
            else:
                points = np.array(json.loads(points_str), dtype=np.float32)
                if points.size == 0:
                    points = np.zeros((0, 2), dtype=np.float32)
                if points.ndim == 1:
                    points = points.reshape(1, 2)
        except Exception as e:
            print(f"Error parsing input_points: {points_str}\nError: {e}")
            points = np.zeros((0, 2), dtype=np.float32)
        try:
            if pd.isna(labels_str) or labels_str.strip() == "":
                labels = np.zeros((0,), dtype=np.int64)
            else:
                labels = np.array(json.loads(labels_str), dtype=np.int64)
        except Exception as e:
            print(f"Error parsing input_labels: {labels_str}\nError: {e}")
            labels = np.zeros((0,), dtype=np.int64)
        return points, labels

    def _pad_points(self, points):
        padded = np.zeros((self.max_points, 2), dtype=np.float32)
        num = min(len(points), self.max_points)
        if num > 0:
            padded[:num] = points[:num]
        return padded

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = row["image_path"]
        mask_path = row["mask_path"]

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        input_points, input_labels = self._parse_points_and_labels(
            row["input_points"], row["input_labels"]
        )

        # Padding to uniform size
        # input_point_padded = self._pad_points(input_points)

        # # Pad input_labels to max_points
        # input_labels = np.array(input_labels, dtype=np.int64)
        # label_padded = np.zeros((self.max_points,), dtype=np.int64)
        # num = min(len(input_labels), self.max_points)
        # if num > 0:
        #     label_padded[:num] = input_labels[:num]

        # # Convert to torch tensors
        # # image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        # # mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        input_points = torch.tensor(input_points, dtype=torch.float32)
        input_labels = torch.tensor(input_labels, dtype=torch.int64)

        return {
            "image": image,
            "mask": mask,
            "input_points": input_points,
            "input_labels": input_labels,
        }
