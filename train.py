from torch.utils.data import DataLoader
from dataloader import SENSATIONDataset
from sam2 import build_sam, sam2_image_predictor
from torch.utils.tensorboard import SummaryWriter
from scripts.transforms import get_training_augmentation
from scripts.utils import read_data
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
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    predictor,
    optimizer,
    scheduler,
    criterion,
    scaler,
    train_data,
    val_data,
    epochs=50,
    accumulation_steps=8,
    steps_per_epoch=1000,
    alpha=0.1,
    device="cuda",
):
    # Set model to training mode
    predictor.model.to(device)
    writer = SummaryWriter(log_dir="logs/training_run_params_test_1")
    # best_val_loss = float("inf")

    # Freeze necessary components (image encoder and prompt encoder)
    for param in predictor.model.sam_prompt_encoder.parameters():
        param.requires_grad = False
    # Assuming image encoder is accessible as predictor.model.image_encoder
    for param in predictor.model.image_encoder.parameters():
        param.requires_grad = False

    # Train Loop
    for epoch in tqdm(range(1, (epochs + 1)), desc="Training Epochs Completed"):

        # Freezing the Prompt encoder
        predictor.model.sam_prompt_encoder.train(True)

        # Enable training for mask decoder
        predictor.model.sam_mask_decoder.train(True)

        ema_loss = None
        epoch_losses = []
        epoch_bce = []
        epoch_dice = []
        epoch_score = []
        epoch_iou = []

        for train_step, data_row in tqdm(
            enumerate(train_data),
            total=steps_per_epoch,
            desc="Training",
        ):
            if train_step >= steps_per_epoch:
                break

            image, mask, input_points, input_labels = read_data(data_row)

            with torch.amp.autocast(device_type="cuda"):
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
                sparse_embeddings, dense_embeddings = (
                    predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, labels),
                        boxes=None,
                        masks=None,
                    )
                )

                batched_mode = unnorm_coords.shape[0] > 1  # multi mask prediction
                high_res_features = [
                    feat_level[-1].unsqueeze(0)
                    for feat_level in predictor._features["high_res_feats"]
                ]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(
                        0
                    ),
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
                gt_mask = torch.from_numpy(mask).float().to(device).unsqueeze(0)

                # prd_mask = prd_masks[:, 0]
                best_scores = torch.argmax(prd_scores, dim=1)
                pred_best_scores = prd_scores[range(len(prd_scores)), best_scores]
                prd_mask = prd_masks[range(len(prd_masks)), best_scores]

                loss, bce, dice, score, iou = criterion(
                    prd_mask, gt_mask, pred_best_scores
                )

                epoch_losses.append(loss.item())
                epoch_bce.append(bce.item())
                epoch_dice.append(dice.item())
                epoch_score.append(score.item())
                epoch_iou.append(iou.item())

                loss = loss / accumulation_steps
                scaler.scale(loss).backward()

            if (train_step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    predictor.model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                predictor.model.zero_grad()
        # Logging of Loss Values both in terminal and Tensorboard (Training Losses)

        mean_loss = sum(epoch_losses) / len(epoch_losses)
        mean_bce = sum(epoch_bce) / len(epoch_bce)
        mean_dice = sum(epoch_dice) / len(epoch_dice)
        mean_score = sum(epoch_score) / len(epoch_score)
        mean_iou = sum(epoch_iou) / len(epoch_iou)
        torch.cuda.synchronize()
        tqdm.write(
            f"Epoch {epoch} | Training Losses -  TotalLoss: {mean_loss:.4f} | BCE: {mean_bce:.4f} | Dice: {mean_dice:.4f} | Score: {mean_score:.4f} | IoU: {mean_iou:.4f},"
        )
        # Logging for Tensorboard and terminal display
        writer.add_scalar("Training/TotalLoss", mean_loss, epoch)
        writer.add_scalar("Training/SigmoidFocal", mean_bce, epoch)
        writer.add_scalar("Training/Dice", mean_dice, epoch)
        writer.add_scalar("Training/Score", mean_score, epoch)
        writer.add_scalar("Training/IoU", mean_iou, epoch)

        # Validation
        val_loss = validation(
            predictor,
            scheduler,
            criterion,
            val_data,
            writer,
            epoch,
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": predictor.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            },
            f"checkpoints/model_checkpoints/sam2_sidewalk_model_checkpoint_{epoch+1}.pt",
        )
        print(f"✅ Saved model at epoch {epoch+1}")


def validation(predictor, scheduler, criterion, val_data, writer, epoch):
    val_losses = []
    val_bce = []
    val_dice = []
    val_score = []
    val_iou = []

    predictor.model.sam_prompt_encoder.eval()
    predictor.model.sam_mask_decoder.eval()

    with torch.no_grad():
        torch.cuda.empty_cache()
        for val_step, val_batch in tqdm(
            enumerate(val_data), total=len(val_data), desc="Validation"
        ):
            image, mask, input_points, input_labels = read_data(val_batch)
            with torch.amp.autocast(device_type="cuda"):
                predictor.set_image(image)
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_points,
                    input_labels,
                    box=None,
                    mask_logits=None,
                    normalize_coords=True,
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
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(
                        0
                    ),
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

                best_scores = torch.argmax(prd_scores, dim=1)
                prd_best_scores = prd_scores[range(len(prd_scores)), best_scores]
                prd_mask = prd_masks[range(len(prd_masks)), best_scores]

                gt_mask = torch.tensor(
                    mask, dtype=torch.float32, device="cuda"
                ).unsqueeze(0)
                prd_mask = torch.sigmoid(prd_masks[:, 0])
                loss_val, bce_val, dice_val, score_val, iou_val = criterion(
                    prd_mask, gt_mask, prd_best_scores
                )
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
        tqdm.write(
            f"Validation Losses - TotalLoss: {avg_loss:.4f} | BCE: {avg_bce:.4f} | Dice: {avg_dice:.4f} | Score: {avg_score:.4f} | IoU: {avg_iou:.4f}"
        )
        writer.add_scalar("Validation/TotalLoss", avg_loss, epoch)
        writer.add_scalar("Validation/SigmoidFocal", avg_bce, epoch)
        writer.add_scalar("Validation/Dice", avg_dice, epoch)
        writer.add_scalar("Validation/Score", avg_score, epoch)
        writer.add_scalar("Validation/IoU", avg_iou, epoch)

        return avg_loss


if __name__ == "__main__":
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 5e-6
    STEP_PER_EPOCH = 2000
    ACCUMULATION_STEPS = 8
    TRAIN_MANIFEST = "dataset/SENSATION_DS_Preprocessed/training/manifest_training.csv"
    VAL_MANIFEST = (
        "dataset/SENSATION_DS_Preprocessed/validation/manifest_validation.csv"
    )
    PRETRAINED_MODEL = "sam2.1_hiera_base+.pt"
    config_file = os.path.abspath("sam2/configs/sam2.1/sam2.1_hiera_b+.yaml")

    augmentations = get_training_augmentation()

    train_dataset = SENSATIONDataset(manifest_path=TRAIN_MANIFEST)
    val_dataset = SENSATIONDataset(manifest_path=VAL_MANIFEST)

    # Initialize model
    sam_model = build_sam.build_sam2(
        checkpoint=PRETRAINED_MODEL, config_file=config_file, device="cuda"
    )
    predictor = sam2_image_predictor.SAM2ImagePredictor(sam_model)
    train_data = DataLoader(train_dataset, shuffle=True)
    val_data = DataLoader(val_dataset, shuffle=True)

    # Define optimizer and scaler
    optimizer = torch.optim.AdamW(
        params=predictor.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=len(train_data) // ACCUMULATION_STEPS, gamma=0.6
    )
    scaler = torch.amp.GradScaler()
    criterion = SegmentationLoss()

    train(
        predictor,
        optimizer,
        scheduler,
        criterion,
        scaler,
        train_data,
        val_data,
        accumulation_steps=ACCUMULATION_STEPS,
        epochs=EPOCHS,
        steps_per_epoch=len(train_data),
    )

    torch.save(predictor.model.state_dict(), "model/sam2_sidewalk_model.pt")
