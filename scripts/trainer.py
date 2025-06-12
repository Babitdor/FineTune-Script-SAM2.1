from scripts.utils import read_data
from scripts.loss import SegmentationLoss
import torch.optim as optim
import os
import torch
import itertools
import gc
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        params,
        predictor,
        train_data,
        val_data,
        device="cuda",
        logging=False,
        tensorboard_logging=True,
    ):
        self.params = params
        self.predictor = predictor
        self.train_data = train_data
        self.val_data = val_data
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.params.gradient_accumulation_steps,
            mixed_precision=(
                "fp16" if self.params.fp16 else ("bf16" if self.params.bf16 else "no")
            ),
        )
        self.logging = logging
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.params.logging_dir)
        self.device = device
        self.tensorboard_logging = tensorboard_logging
        self.global_step = 0
        self.best_metric = (
            -float("inf") if self.params.greater_is_better else float("inf")
        )
        self.checkpoint_history = []

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            params=self.predictor.model.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
        )
        # Prepare components with accelerator
        self.predictor.model, self.optimizer, self.train_data, self.val_data = (
            self.accelerator.prepare(
                self.predictor.model, self.optimizer, self.train_data, self.val_data
            )
        )

        # Create infinite data iterator
        self.train_data_iter = itertools.cycle(self.train_data)

    def train(self):
        total_steps = self.params.total_steps
        pbar = tqdm(
            total=total_steps, desc="Training Progress", initial=self.global_step
        )

        # Warmup scheduler
        warmup_scheduler = self.get_warmup_scheduler()

        while self.global_step < total_steps:
            self.train_step(pbar, warmup_scheduler)

        self.save_final_model()
        pbar.close()
        self.writer.close()

    def get_warmup_scheduler(self):
        """Creates a linear warmup scheduler"""

        def warmup_lr_scheduler(step):
            if step < self.params.warmup_steps:
                return float(step) / float(max(1, self.params.warmup_steps))
            return 1.0

        return warmup_lr_scheduler

    def train_step(self, pbar, warmup_scheduler):

        # Get next batch
        data_row = next(self.train_data_iter)

        # Set model to training mode
        self.predictor.model.sam_prompt_encoder.train(
            bool(self.params.train_prompt_encoder)
        )
        self.predictor.model.sam_mask_decoder.train(
            bool(self.params.train_mask_decoder)
        )

        # Prepare data
        image, mask, input_points, input_labels = read_data(data_row)

        # Forward pass with mixed precision
        with self.accelerator.accumulate(self.predictor.model):
            with torch.amp.autocast(device_type=self.device.type):
                self.predictor.set_image(image)
                mask_input, unnorm_coords, labels, unnorm_box = (
                    self.predictor._prep_prompts(
                        input_points,
                        input_labels,
                        box=None,
                        mask_logits=None,
                        normalize_coords=True,
                    )
                )

                # Calculate embeddings
                sparse_embeddings, dense_embeddings = (
                    self.predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, labels),
                        boxes=None,
                        masks=None,
                    )
                )

                # Model prediction
                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [
                    feat_level[-1].unsqueeze(0)
                    for feat_level in self.predictor._features["high_res_feats"]
                ]
                low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                    image_embeddings=self.predictor._features["image_embed"][
                        -1
                    ].unsqueeze(0),
                    image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )

                # Process masks
                prd_masks = self.predictor._transforms.postprocess_masks(
                    low_res_masks, self.predictor._orig_hw[-1]
                )
                prd_masks = torch.sigmoid(prd_masks.squeeze(0))
                gt_mask = mask.unsqueeze(0)

                # Calculate loss
                loss, iou = SegmentationLoss()(prd_masks, gt_mask, prd_scores)

                # Apply gradient accumulation
                loss = loss / self.params.gradient_accumulation_steps

            # Backward pass
            self.accelerator.backward(loss)

            # Gradient clipping and optimization step
            if self.accelerator.sync_gradients:
                # Clip gradients
                self.accelerator.clip_grad_norm_(
                    self.predictor.model.parameters(), max_norm=1.0
                )

            # Apply LR warmup
            if self.global_step < self.params.warmup_steps:
                warmup_factor = warmup_scheduler(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.params.learning_rate * warmup_factor

                # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Update global step
        self.global_step += 1
        pbar.update(1)

        # Calculate fractional epoch
        fractional_epoch = (
            self.global_step
            * int(self.params.per_device_train_batch_size)
            / len(self.train_data.dataset)
        )

        # Logging
        if self.global_step % self.params.logging_steps == 0 and self.logging:
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"Step {self.global_step} | Epoch {fractional_epoch:.2f} | "
                f"Loss: {loss.item() * self.params.gradient_accumulation_steps:.4f} | "
                f"IoU: {iou.mean().item():.4f} | LR: {current_lr:.2e}"
            )

            if self.accelerator.is_local_main_process and self.tensorboard_logging:
                # TensorBoard logging
                self.writer.add_scalar(
                    "train/loss",
                    loss.item() * self.params.gradient_accumulation_steps,
                    self.global_step,
                )
                self.writer.add_scalar("train/iou", iou.mean().item(), self.global_step)
                self.writer.add_scalar("train/lr", current_lr, self.global_step)

        # Evaluation
        if (
            self.params.eval_strategy == "steps"
            and self.global_step % self.params.eval_steps == 0
        ):
            eval_metrics = self.validate()
            eval_iou = eval_metrics["iou"]

            # Save best model
            if self.params.load_best_model_at_end:
                is_best = (
                    (eval_iou > self.best_metric)
                    if self.params.greater_is_better
                    else (eval_iou < self.best_metric)
                )
                if is_best:
                    self.best_metric = eval_iou
                    self.save_checkpoint(self.global_step, is_best=True)

        # Save checkpoint
        if (
            self.params.save_strategy == "steps"
            and self.global_step % self.params.save_steps == 0
        ):
            self.save_checkpoint(self.global_step)

        # Commented Code : For Memory Leaks Monitoring
        # if torch.cuda.is_available():
        #     print(
        #         f"[BEFORE - Step {self.global_step}] CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        #     )

        # Clearing out tensors and reseting for memory management
        del image, mask, input_points, input_labels
        del mask_input, unnorm_coords, labels, unnorm_box
        del sparse_embeddings, dense_embeddings
        del low_res_masks, prd_scores, prd_masks, gt_mask
        del loss, iou
        self.cleanup_memory(step=self.global_step)

        # Commented Code : For Memory Leaks Monitoring
        # if torch.cuda.is_available():
        #     print(
        #         f"[AFTER - Step {self.global_step}] CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        #     )

    def cleanup_memory(self, step):
        """Aggressive memory cleanup to prevent slowdowns"""
        # 1. Reset predictor state
        self.predictor.reset_predictor()
        # 2. Explicitly delete tensors that might be lingering
        if hasattr(self, "image"):
            del self.image
        if hasattr(self, "mask"):
            del self.mask
        if hasattr(self, "input_points"):
            del self.input_points
        if hasattr(self, "input_labels"):
            del self.input_labels

        # 3. Force Python garbage collection
        gc.collect()

        # 4. Clear GPU cache more frequently
        torch.cuda.empty_cache()

    def validate(self):
        self.predictor.model.sam_prompt_encoder.eval()
        self.predictor.model.sam_mask_decoder.eval()

        val_losses = []
        val_ious = []

        with torch.no_grad():
            for val_row in tqdm(
                self.val_data, desc=f"Evaluating Step {self.global_step}", leave=False
            ):
                # Prepare data
                image, mask, input_points, input_labels = read_data(val_row)

                # Forward pass
                with torch.amp.autocast(
                    device_type=self.device.type, enabled=self.params.fp16
                ):
                    self.predictor.set_image(image)
                    mask_input, unnorm_coords, labels, unnorm_box = (
                        self.predictor._prep_prompts(
                            input_points,
                            input_labels,
                            box=None,
                            mask_logits=None,
                            normalize_coords=True,
                        )
                    )

                    # Calculate embeddings
                    sparse_embeddings, dense_embeddings = (
                        self.predictor.model.sam_prompt_encoder(
                            points=(unnorm_coords, labels),
                            boxes=None,
                            masks=None,
                        )
                    )

                    # Model prediction
                    batched_mode = unnorm_coords.shape[0] > 1
                    high_res_features = [
                        feat_level[-1].unsqueeze(0)
                        for feat_level in self.predictor._features["high_res_feats"]
                    ]
                    low_res_masks, prd_scores, _, _ = (
                        self.predictor.model.sam_mask_decoder(
                            image_embeddings=self.predictor._features["image_embed"][
                                -1
                            ].unsqueeze(0),
                            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            repeat_image=batched_mode,
                            high_res_features=high_res_features,
                        )
                    )

                    # Process masks
                    prd_masks = self.predictor._transforms.postprocess_masks(
                        low_res_masks, self.predictor._orig_hw[-1]
                    )
                    prd_masks = torch.sigmoid(prd_masks.squeeze(0))
                    gt_mask = mask.unsqueeze(0)

                    # Calculate loss and IoU
                    loss_val, iou_val = SegmentationLoss()(
                        prd_masks, gt_mask, prd_scores
                    )

                # Accumulate validation stats
                val_losses.append(loss_val.item())
                val_ious.append(iou_val.mean().item())

                # Clearing out tensors and reseting for memory management
                self.predictor.reset_predictor()
                del image, mask, input_points, input_labels
                del mask_input, unnorm_coords, labels, unnorm_box
                del sparse_embeddings, dense_embeddings
                del low_res_masks, prd_scores, prd_masks, gt_mask
                del loss_val, iou_val
                gc.collect()

                torch.cuda.empty_cache()

        # Calculate validation metrics
        metrics = {
            "loss": sum(val_losses) / len(val_losses),
            "iou": sum(val_ious) / len(val_ious),
        }

        if self.accelerator.is_local_main_process:
            # TensorBoard logging
            self.writer.add_scalar("val/loss", metrics["loss"], self.global_step)
            self.writer.add_scalar("val/iou", metrics["iou"], self.global_step)

            tqdm.write(
                f"Step {self.global_step} | Validation | "
                f"Loss: {metrics['loss']:.4f} | IoU: {metrics['iou']:.4f}"
            )

            torch.cuda.empty_cache()

            # Commented Code : For Memory Leaks Monitoring
            # if torch.cuda.is_available():
            #     print(
            #         f"[Validation Step {self.global_step}] CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            #     )

        return metrics

    def save_checkpoint(self, step, is_best=False):
        if not self.accelerator.is_local_main_process:
            return

        # Create output directory
        os.makedirs(os.path.join(self.params.output_dir, "checkpoint"), exist_ok=True)

        # Prepare checkpoint
        checkpoint = {
            "step": step,
            "model_state_dict": self.accelerator.get_state_dict(self.predictor.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "params": self.params.to_dict(),
        }

        # Save checkpoint
        checkpoint_path = os.path.join(
            self.params.output_dir, "checkpoint", f"checkpoint_step_{step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Track checkpoint history
        self.checkpoint_history.append((step, checkpoint_path))

        # Save as best if applicable
        if is_best:
            best_path = os.path.join(self.params.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"\nSaved best model at step {step} with IoU: {self.best_metric:.4f}")

        # Apply save total limit
        if len(self.checkpoint_history) > self.params.save_total_limit:
            # Sort by step (oldest first)
            self.checkpoint_history.sort(key=lambda x: x[0])

            # Remove oldest checkpoint
            _, oldest_path = self.checkpoint_history.pop(0)
            if os.path.exists(oldest_path):
                os.remove(oldest_path)

        del checkpoint
        gc.collect()

        # Commented Code : For Memory Leaks Monitoring
        # if torch.cuda.is_available():
        #     print(
        #         f"[Checkpoint Step {step}] CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        #     )

    def save_final_model(self):
        if not self.accelerator.is_local_main_process:
            return

        # Create output directory
        os.makedirs(self.params.output_dir, exist_ok=True)

        # Save final model
        final_path = os.path.join(self.params.output_dir, "final_model.pt")
        torch.save(self.accelerator.get_state_dict(self.predictor.model), final_path)
        print("✅ Final model saved")

        # Save best model if configured
        if self.params.load_best_model_at_end:
            best_path = os.path.join(self.params.output_dir, "best_model.pt")
            if os.path.exists(best_path):
                print(f"Best model saved with IoU: {self.best_metric:.4f}")
            else:
                # If no best model saved during training, save current as best
                torch.save(
                    self.accelerator.get_state_dict(self.predictor.model), best_path
                )
                print("✅ Best model saved (final model)")
