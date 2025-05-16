from torch.utils.data import DataLoader
from dataloader import SENSATIONDataset
from sam2 import build_sam, sam2_image_predictor
from torch.utils.tensorboard import SummaryWriter
from loss import SegmentationLoss
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import torch
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
writer = SummaryWriter(log_dir="logs/training_run_1")


def read_data(data):
    for batch in data:
        image = batch["image"][0]  # shape: [C, H, W] or [H, W, C]
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        mask = batch["mask"][0]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        input_points = batch["input_points"][0]
        input_labels = batch["input_labels"][0]
        return image, mask, input_points, input_labels


def train(
    predictor,
    optimizer,
    scheduler,
    criterion,
    scalar,
    train_data,
    val_data,
    epochs=10000,
    steps=100,
    accumulation_steps=4,
    log_interval=1000,
):
    # Set model to training mode
    predictor.model.train()
    # Train Loop
    for step in tqdm(range(1, epochs + 1), desc="Training"):
        with torch.amp.autocast(device_type="cuda"):
            image, mask, input_points, input_labels = read_data(train_data)

            # Setting Image and Preparing prompts
            predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                input_points,
                input_labels,
                box=None,
                mask_logits=None,
                normalize_coords=True,
            )

            # Calculating prompting embeddings
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels),
                boxes=None,
                masks=None,
            )

            batched_mode = unnorm_coords.shape[0] > 1  # multi mask prediction
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

            # Upscales the masks to the original image resolution
            prd_masks = predictor._transforms.postprocess_masks(
                low_res_masks, predictor._orig_hw[-1]
            )

            # Loss Calculations (Training)
            gt_mask = torch.tensor(mask, dtype=torch.float32, device="cuda").unsqueeze(
                0
            )
            prd_mask = torch.sigmoid(prd_masks[:, 0])
            loss, bce, dice, score, iou = criterion(prd_mask, gt_mask, prd_scores)

        loss = loss / accumulation_steps
        scalar.scale(loss).backward()
        if (step + 1) % accumulation_steps == 0:
            scalar.step(optimizer)
            scalar.update()
            predictor.model.zero_grad()

        # Logging for Tensorboard and terminal display
        writer.add_scalar("TotalLoss", loss.item(), step)
        writer.add_scalar("Loss/BCE", bce.item(), step)
        writer.add_scalar("Loss/Dice", dice.item(), step)
        writer.add_scalar("Loss/Score", score.item(), step)
        writer.add_scalar("Metric/IoU", iou.item(), step)

        if step % log_interval == 0:

            tqdm.write(
                f"Training Losses -  TotalLoss: {loss.item():.4f} | BCE: {bce.item():.4f} | Dice: {dice.item():.4f} | Score: {score.item():.4f} | IoU: {iou.item():.4f},"
            )
            predictor.model.eval()

            val_losses = []
            val_bce = []
            val_dice = []
            val_score = []
            val_iou = []

            with torch.no_grad():
                for val_step in tqdm(range(1, steps + 1), desc="Validation"):
                    image, mask, input_points, input_labels = read_data(val_data)
                    predictor.set_image(image)
                    mask_input, unnorm_coords, labels, unnorm_box = (
                        predictor._prep_prompts(
                            input_points,
                            input_labels,
                            box=None,
                            mask_logits=None,
                            normalize_coords=True,
                        )
                    )

                    sparse_embeddings, dense_embeddings = (
                        predictor.model.sam_prompt_encoder(
                            points=(unnorm_coords, labels),
                            boxes=None,
                            masks=None,
                        )
                    )

                    batched_mode = unnorm_coords.shape[0] > 1
                    high_res_features = [
                        feat_level[-1].unsqueeze(0)
                        for feat_level in predictor._features["high_res_feats"]
                    ]
                    low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                        image_embeddings=predictor._features["image_embed"][
                            -1
                        ].unsqueeze(0),
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

                    gt_mask = torch.tensor(
                        mask, dtype=torch.float32, device="cuda"
                    ).unsqueeze(0)
                    prd_mask = torch.sigmoid(prd_masks[:, 0])
                    loss_val, bce_val, dice_val, score_val, iou_val = criterion(
                        prd_mask, gt_mask, prd_scores
                    )
                    if val_step == 1:
                        plt.figure(figsize=(8, 4))
                        plt.subplot(1, 2, 1)
                        plt.title("Ground Truth Mask")
                        plt.imshow(
                            gt_mask.cpu().detach().numpy().squeeze(), cmap="gray"
                        )
                        plt.axis("off")
                        plt.subplot(1, 2, 2)
                        plt.title("Predicted Mask")
                        plt.imshow(
                            ((prd_mask > 0.5).float()).cpu().detach().numpy().squeeze(),
                            cmap="gray",
                        )
                        plt.axis("off")
                        plt.tight_layout()
                        plt.show()

                    val_losses.append(loss_val.item())
                    val_bce.append(bce_val.item())
                    val_dice.append(dice_val.item())
                    val_score.append(score_val.item())
                    val_iou.append(iou_val.item())

            # Aggregate validation metrics
            avg_loss = sum(val_losses) / len(val_losses)
            avg_bce = sum(val_bce) / len(val_bce)
            avg_dice = sum(val_dice) / len(val_dice)
            avg_score = sum(val_score) / len(val_score)
            avg_iou = sum(val_iou) / len(val_iou)

            # Log aggregated validation metrics for the current epoch/step
            writer.add_scalar("Validation/Total_Loss", avg_loss, step)
            writer.add_scalar("Validation/BCE", avg_bce, step)
            writer.add_scalar("Validation/Dice", avg_dice, step)
            writer.add_scalar("Validation/Score", avg_score, step)
            writer.add_scalar("Validation/IoU", avg_iou, step)
            tqdm.write(
                f"Validation Losses - TotalLoss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | Dice: {avg_dice:.4f} | Score: {avg_score:.4f} | IoU: {avg_iou:.4f}"
            )

            scheduler.step(avg_loss)
            # Return model to training mode after validation
            predictor.model.train()


if __name__ == "__main__":
    BATCH_SIZE = 1
    EPOCHS = 10000
    STEPS = 100
    ACCUMULATION_STEPS = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    LOG_INTERVALS = 1000
    config_file = os.path.abspath("sam2/configs/sam2.1/sam2.1_hiera_s.yaml")

    train_dataset = SENSATIONDataset(
        manifest_path="dataset/SENSATION_DS_Preprocessed/training/manifest_training.csv",
        root_dir="",
        max_points=10,
    )
    val_dataset = SENSATIONDataset(
        manifest_path="dataset/SENSATION_DS_Preprocessed/validation/manifest_validation.csv",
        root_dir="",
        max_points=10,
    )

    # Initialize model
    sam_model = build_sam.build_sam2(
        checkpoint="sam2.1_hiera_small.pt",
        config_file=config_file,
    )
    predictor = sam2_image_predictor.SAM2ImagePredictor(sam_model)
    train_data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define optimizer and scaler
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    scalar = torch.amp.GradScaler()
    criterion = SegmentationLoss()

    train(
        predictor,
        optimizer,
        scheduler,
        criterion,
        scalar,
        train_data,
        val_data,
        epochs=EPOCHS,
        steps=STEPS,
        accumulation_steps=ACCUMULATION_STEPS,
        log_interval=LOG_INTERVALS,
    )
