import os
import cv2
import json
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch


class SENSATIONDataset(Dataset):
    def __init__(self, manifest_path, transform=None):
        self.data = pd.read_csv(manifest_path)
        self.transform = transform

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return image

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        return mask

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image = self._load_image(row["image_path"])
        mask = self._load_mask(row["mask_path"])

        if self.transform is not None:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image = transformed["image"]
            mask = transformed["mask"] / 255
        else:
            image = np.array(image)
            mask = np.array(mask) / 255

        input_points, input_labels = self._parse_points_and_labels(
            row["input_points"], row["input_labels"]
        )

        # Convert to torch tensors
        input_points = torch.tensor(input_points)
        input_labels = torch.tensor(input_labels)

        return {
            "image": image,
            "mask": mask,
            "input_points": input_points,
            "input_labels": input_labels,
        }
