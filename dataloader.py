import os
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
        self.transfrom = transform
        self.max_points = max_points

    def _load_image(self, path):
        image = Image.open(path).convert("RGB")
        return np.array(image)

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        return np.array(mask) // 255

    def _parse_points(self, point_str):
        if pd.isna(point_str) or point_str.strip() == "":
            return np.zeros((0, 2), dtype=np.float32)
        arr = np.array(json.loads(point_str), dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)
        return arr

    def _pad_points(self, points):
        padded = np.zeros((self.max_points, 2), dtype=np.float32)
        num = min(len(points), self.max_points)
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

        pos_points = self._parse_points(row["positive_points"])
        neg_points = self._parse_points(row["negative_points"])

        # Padding to uniform size
        pos_padded = self._pad_points(pos_points)
        neg_padded = self._pad_points(neg_points)

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        pos_padded = torch.tensor(pos_padded, dtype=torch.float32)
        neg_padded = torch.tensor(neg_padded, dtype=torch.float32)

        return {
            "image": image,
            "mask": mask,
            "positive_points": pos_padded,
            "negative_points": neg_padded,
        }
