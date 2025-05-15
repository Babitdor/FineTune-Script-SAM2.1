from torch.utils.data import DataLoader
from dataloader import SENSATIONDataset
import os

dataset = SENSATIONDataset(
    manifest_path="dataset/SENSATION_DS_Preprocessed/training/manifest_training.csv",
    root_dir="",
    max_points=10,
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in loader:
    print(batch["image"].shape)
    print(batch["positive_points"].shape)
