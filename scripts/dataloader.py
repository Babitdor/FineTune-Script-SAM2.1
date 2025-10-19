import os
from torch.utils.data import Dataset


class SENSATIONDataset(Dataset):
    def __init__(self, root_dir, split="training"):
        """
        Args:
            root_dir (string): Root directory of the dataset (contains 'training' and 'validation' folders)
            split (string): Either 'training' or 'validation'
        """
        self.root_dir = root_dir
        self.split = split

        # Set paths for images and masks
        self.image_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")

        # Get list of image files (assuming masks have same names)
        self.image_files = sorted(
            [
                f
                for f in os.listdir(self.image_dir)
                if os.path.isfile(os.path.join(self.image_dir, f))
            ]
        )

        # Verify that corresponding masks exist
        self.valid_pairs = []
        for img_file in self.image_files:
            mask_file = os.path.splitext(img_file)[0] + ".png"
            mask_path = os.path.join(self.mask_dir, mask_file)

            if os.path.exists(mask_path):
                self.valid_pairs.append((img_file, mask_file))
            else:
                print(f"Warning: No corresponding mask found for image {img_file}")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        return {"image_path": image_path, "mask_path": mask_path}
