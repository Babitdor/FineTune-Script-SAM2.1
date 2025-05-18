import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloader import SENSATIONDataset
from sam2 import build_sam, sam2_image_predictor
import random
from scripts.utils import post_process

# Paths
MODEL_PATH = "model/sam2_final_model.pt"
CONFIG_PATH = r"D:\FAU\Semester_3\Project II\sam2\sidewalk_training\sam2\configs\sam2.1\sam2.1_hiera_b+.yaml"
TEST_MANIFEST = "dataset/SENSATION_DS_Preprocessed/training/manifest_training.csv"

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = build_sam.build_sam2(
    checkpoint=None,  # No checkpoint, just weights
    config_file=CONFIG_PATH,
)
sam_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
sam_model.to(device)
sam_model.eval()

# Predictor
predictor = sam2_image_predictor.SAM2ImagePredictor(sam_model)

# Dataset
test_dataset = SENSATIONDataset(manifest_path=TEST_MANIFEST)
# Use batch_size=1 for inference
from torch.utils.data import DataLoader

test_loader = DataLoader(test_dataset, shuffle=False)

# Inference loop
with torch.no_grad():
    # Get all indices and shuffle them
    indices = list(range(len(test_dataset)))
    random.shuffle(indices)
    # Show 10 random samples
    for shown, idx in enumerate(indices):
        batch = test_dataset[idx]
        image = batch["image"]
        mask = batch["mask"]
        input_points = batch["input_points"]
        input_labels = batch["input_labels"]

        # Convert image to numpy HWC uint8 if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_points,
            input_labels,
            box=None,
            mask_logits=None,
            normalize_coords=True,
        )
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels),
            boxes=None,
            masks=None,
        )
        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [
            feat_level[-1].unsqueeze(0)
            for feat_level in predictor._features["high_res_feats"]
        ]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(
            low_res_masks, predictor._orig_hw[-1]
        )
        prd_mask = prd_masks[:, 0].cpu().numpy().squeeze()
        pred_bin_mask_pro = post_process(
            prd_mask, thresh=0.7, open_k=3, close_k=3, smooth=False, iterations=1
        )

        # Visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask.squeeze(), cmap="gray")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_bin_mask_pro, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        if shown >= 9:
            break
