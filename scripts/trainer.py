import shutil

from scripts.load_data import read_batch
from scripts.loss import SegmentationLoss
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
import os
import numpy as np
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
        model_name,
        device="cuda",
        logging=False,
        tensorboard_logging=True,
        visualize=False,
    ):
        self.params = params
        self.predictor = predictor

        self.configure_training(params=params)
        self.train_data = train_data
        self.val_data = val_data
        self.model_name = model_name.replace(".yaml", "")

        flags = []
        if self.params.use_lora:
            flags.append("lora")
            if self.params.lora_image_encoder:
                flags.append("i")
            if self.params.lora_mask_decoder:
                flags.append("d")

        else:
            flags.append("finetune")
            if self.params.train_mask_decoder:
                flags.append("d")
            if self.params.train_prompt_encoder:
                flags.append("p")
            if self.params.train_image_encoder:
                flags.append("i")

        self.name_suffix = "_".join(flags) or ""
        self.visualize = visualize
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.params.gradient_accumulation_steps,
            mixed_precision=(
                "fp16" if self.params.fp16 else ("bf16" if self.params.bf16 else "no")
            ),
        )
        self.logging = logging
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                self.params.logging_dir, f"{self.name_suffix}_logs", self.model_name
            )
        )
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

    def configure_training(self, params):

        if getattr(params, "use_lora", False):
            target_modules = []
            # If using LoRA, apply LoRA to targeted layers
            if getattr(params, "lora_image_encoder", False):
                target_modules.extend(
                    [
                        "trunk.blocks.*.attn.qkv",  # Attention QKV projection
                        "trunk.blocks.*.attn.proj",  # Attention output projection
                        "trunk.blocks.*.mlp.layers.0",  # MLP first linear layer
                        "trunk.blocks.*.mlp.layers.1",  # MLP second linear layer
                        "trunk.blocks.1.proj",  # Projection in block 1
                        "trunk.blocks.3.proj",  # Projection in block 3
                        "trunk.blocks.10.proj",  # Projection in block 10
                    ]
                )
            if getattr(params, "lora_mask_decoder", False):
                target_modules.extend(
                    [
                        "transformer.layers.*.self_attn.q_proj",
                        "transformer.layers.*.self_attn.k_proj",
                        "transformer.layers.*.self_attn.v_proj",
                        "transformer.layers.*.self_attn.out_proj",
                        "transformer.layers.*.cross_attn_token_to_image.q_proj",
                        "transformer.layers.*.cross_attn_token_to_image.k_proj",
                        "transformer.layers.*.cross_attn_token_to_image.v_proj",
                        "transformer.layers.*.cross_attn_token_to_image.out_proj",
                        "transformer.layers.*.cross_attn_image_to_token.q_proj",
                        "transformer.layers.*.cross_attn_image_to_token.k_proj",
                        "transformer.layers.*.cross_attn_image_to_token.v_proj",
                        "transformer.layers.*.cross_attn_image_to_token.out_proj",
                        "transformer.layers.*.mlp.layers.0",
                        "transformer.layers.*.mlp.layers.1",
                        "transformer.final_attn_token_to_image.q_proj",
                        "transformer.final_attn_token_to_image.k_proj",
                        "transformer.final_attn_token_to_image.v_proj",
                        "transformer.final_attn_token_to_image.out_proj",
                        "output_hypernetworks_mlps.*.layers.0",
                        "output_hypernetworks_mlps.*.layers.1",
                        "output_hypernetworks_mlps.*.layers.2",
                        "iou_prediction_head.layers.0",
                        "iou_prediction_head.layers.1",
                        "iou_prediction_head.layers.2",
                        "pred_obj_score_head.layers.0",
                        "pred_obj_score_head.layers.1",
                        "pred_obj_score_head.layers.2",
                    ]
                )
            # Check the targetted modules for LoRA
            # print(target_modules)

            if target_modules:
                lora_config = LoraConfig(
                    r=params.lora_r,
                    lora_alpha=params.lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=params.lora_dropout,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                self.predictor.model = get_peft_model(self.predictor.model, lora_config)
                self.predictor.model.print_trainable_parameters()
                print("LoRA Fine Tuning Activated")
            else:
                print(
                    "Warning: No LoRA target modules specified (lora_image_encoder and lora_mask_decoder are False)."
                )
            # Set training mode for sam_mask_decoder (for dropout in LoRA or transformer)
            self.predictor.model.sam_mask_decoder.train(getattr(params, "lora_mask_decoder", False))  # type: ignore
            self.predictor.model.image_encoder.train(getattr(params, "lora_image_encoder", False))  # type: ignore
            self.predictor.model.sam_prompt_encoder.eval()  # type: ignore

        else:
            # Full fine-tuning: unfreeze modules based on params
            if params.train_mask_decoder:
                for param in self.predictor.model.sam_mask_decoder.parameters():
                    param.requires_grad = True
                self.predictor.model.sam_mask_decoder.train(True)
            else:
                self.predictor.model.sam_mask_decoder.eval()

            if params.train_prompt_encoder:
                for param in self.predictor.model.sam_prompt_encoder.parameters():
                    param.requires_grad = True
                self.predictor.model.sam_prompt_encoder.train(True)
            else:
                self.predictor.model.sam_prompt_encoder.eval()

            if params.train_image_encoder:
                for param in self.predictor.model.image_encoder.parameters():
                    param.requires_grad = True
                self.predictor.model.image_encoder.train(True)
            else:
                self.predictor.model.image_encoder.eval()
            print("Default Full Fine Tuning Activated")

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
        """Creates a linear warmup scheduler followed by cosine annealing"""

        def warmup_cosine_lr_scheduler(step):
            if step < self.params.warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, self.params.warmup_steps))
            else:
                # Cosine annealing from learning_rate to min_lr
                min_lr = getattr(self.params, "min_lr", 1e-6)
                decay_steps = self.params.total_steps - self.params.warmup_steps
                current_step = step - self.params.warmup_steps
                cosine_decay = 0.5 * (1.0 + np.cos(np.pi * current_step / decay_steps))
                return (
                    cosine_decay
                    * (self.params.learning_rate - min_lr)
                    / self.params.learning_rate
                    + min_lr / self.params.learning_rate
                )

        return warmup_cosine_lr_scheduler

    def train_step(self, pbar, warmup_scheduler):

        # Prepare data
        image, mask, input_points, input_labels = read_batch(
            data=self.train_data,
            strategy=self.params.point_strategy,
            num_pts=self.params.num_pts,
            visualize=self.visualize,
        )

        # Forward pass with mixed precision
        with self.accelerator.accumulate(self.predictor.model):  # type: ignore
            with torch.amp.autocast(device_type=self.device.type):  # type: ignore
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
                batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
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
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )

                prd_masks = self.predictor._transforms.postprocess_masks(
                    low_res_masks, self.predictor._orig_hw[-1]
                )  # Upscale the masks to the original image resolution

                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                prd_masks = torch.sigmoid(prd_masks[:, 0])

                # Calculate loss
                loss, iou = SegmentationLoss(
                    score_loss_weight=self.params.score_weight
                )(prd_masks, gt_mask, prd_scores)

                # Apply gradient accumulation
                loss = loss / self.params.gradient_accumulation_steps

            # Backward pass
            self.accelerator.backward(loss)

            # Apply LR warmup
            warmup_factor = warmup_scheduler(self.global_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.params.learning_rate * warmup_factor

            # Gradient clipping and optimization step
            if self.accelerator.sync_gradients:
                # Clip gradients
                self.accelerator.clip_grad_norm_(
                    self.predictor.model.parameters(), max_norm=1.0
                )
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Update global step
        self.global_step += 1
        pbar.update(1)

        # Calculate fractional epoch
        # fractional_epoch = (
        #     self.global_step
        #     * int(self.params.per_device_train_batch_size)
        #     / len(self.train_data.dataset)
        # )

        # Logging
        if self.global_step % self.params.logging_steps == 0 and self.logging:
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_description(
                f"Step {self.global_step} | Training | "
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
        self.cleanup_memory()

        # Commented Code : For Memory Leaks Monitoring
        # if torch.cuda.is_available():
        #     print(
        #         f"[AFTER - Step {self.global_step}] CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        #     )

    def cleanup_memory(self):
        """Aggressive memory cleanup to prevent slowdowns"""
        # 1. Reset predictor state
        self.predictor.reset_predictor()
        # 2. Explicitly delete tensors that might be lingering
        if hasattr(self, "image"):
            del self.image  # type: ignore
        if hasattr(self, "mask"):
            del self.mask  # type: ignore
        if hasattr(self, "input_points"):
            del self.input_points  # type: ignore
        if hasattr(self, "input_labels"):
            del self.input_labels  # type: ignore

        # 3. Force Python garbage collection
        gc.collect()

        # 4. Clear GPU cache more frequently
        torch.cuda.empty_cache()

    def validate(self):
        self.predictor.model.eval()

        val_losses = []
        val_ious = []

        with torch.no_grad():
            for val_row in tqdm(
                self.val_data, desc=f"Evaluating Step {self.global_step}", leave=False
            ):
                # Prepare data
                image, mask, input_points, input_labels = read_batch(
                    data=self.val_data,
                    strategy=self.params.point_strategy,
                    num_pts=self.params.num_pts,
                )

                # Forward pass
                with torch.amp.autocast(  # type: ignore
                    device_type=self.device.type, enabled=self.params.fp16  # type: ignore
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
                    batched_mode = unnorm_coords.shape[0] > 1  # multi object prediction
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
                            multimask_output=True,
                            repeat_image=batched_mode,
                            high_res_features=high_res_features,
                        )
                    )

                    prd_masks = self.predictor._transforms.postprocess_masks(
                        low_res_masks, self.predictor._orig_hw[-1]
                    )  # Upscale the masks to the original image resolution

                    gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                    prd_masks = torch.sigmoid(prd_masks[:, 0])

                    # Calculate loss and IoU
                    loss_val, iou_val = SegmentationLoss(
                        score_loss_weight=self.params.score_weight
                    )(prd_masks, gt_mask, prd_scores)

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

        self.checkpts_subpath = f"{self.name_suffix}_checkpoint/{self.model_name}"

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
            self.params.output_dir,
            self.checkpts_subpath,
        )
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, checkpoint_file)

        # Track checkpoint history
        self.checkpoint_history.append((step, checkpoint_file))

        # Save as best if applicable
        if is_best:
            best_model_metadata = {
                "config": vars(self.params),  # Optional: save training config
                "metadata": {
                    "best_iou": self.best_metric,
                    "save_type": "lora" if self.params.use_lora else "full",
                },
            }
            if self.params.use_lora:

                best_path = os.path.join(
                    self.params.output_dir,
                    self.checkpts_subpath,
                    "LoRA_Adapters",
                    "best_adapters",
                )
                os.makedirs(best_path, exist_ok=True)
                self.accelerator.unwrap_model(self.predictor.model).save_pretrained(
                    best_path
                )
                self.accelerator.save(
                    best_model_metadata,
                    os.path.join(best_path, f"best_model_metadata.pt"),
                )
                tqdm.write(f"✅ Saved best LoRA adapters at {best_path}")
                tqdm.write(f"✅ Saved at step: {step} with IOU: {self.best_metric:.4f}")

            else:
                best_path = os.path.join(
                    self.params.output_dir,
                    self.checkpts_subpath,
                    "Fine_tuned",
                )
                os.makedirs(best_path, exist_ok=True)
                torch.save(
                    {
                        "model": self.accelerator.get_state_dict(self.predictor.model),
                        **best_model_metadata,
                    },
                    os.path.join(
                        best_path,
                        f"{self.model_name}_best_model.pt",
                    ),
                )
                tqdm.write(f"✅ Saved best Fine_tuned_Models saved at {best_path}")
                tqdm.write(f"✅ Saved at step: {step} with IOU: {self.best_metric:.4f}")

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

    def save_final_model(self):
        if not self.accelerator.is_local_main_process:
            return

        os.makedirs(self.params.output_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        # Common model metadata
        metadata = {
            "config": self.params.to_dict(),
            "metadata": {
                "best_iou": self.best_metric,
                "step": self.global_step,
                "save_type": "lora" if self.params.use_lora else "full",
            },
        }

        if self.params.use_lora:
            # Save LoRA adapters
            adapter_dir = os.path.join(
                self.params.output_dir,
                self.checkpts_subpath,
                "LoRA_Adapters",
                "final_adapters",
            )
            os.makedirs(adapter_dir, exist_ok=True)
            self.accelerator.unwrap_model(self.predictor.model).save_pretrained(
                adapter_dir
            )
            self.accelerator.save(
                metadata,
                os.path.join(
                    adapter_dir,
                    "final_model_metadata.pt",
                ),
            )
            tqdm.write(f"✅ Final LoRA adapters saved at {adapter_dir}")
        else:
            # Save full fine tune model
            final_path = os.path.join(
                self.params.output_dir,
                self.checkpts_subpath,
                "Fine_tuned",
            )
            os.makedirs(final_path, exist_ok=True)
            torch.save(
                {
                    "model": self.accelerator.get_state_dict(self.predictor.model),
                    **metadata,
                },
                os.path.join(final_path, f"{self.model_name}_final_model.pt"),
            )
            tqdm.write(f"✅ Full model saved to {final_path}")

        # Handle best model saving
        # if self.params.load_best_model_at_end:
        #     best_path = os.path.join(
        #         self.params.output_dir,
        #         f"{self.model_name}_best_model_{self.name_suffix}.pt",
        #     )
        #     best_model = {
        #         "model": self.accelerator.get_state_dict(self.predictor.model),
        #         "config": self.params.to_dict(),
        #         "metadata": {"best_iou": self.best_metric, "step": self.global_step},
        #     }
        #     self.accelerator.save(best_model, best_path)
        #     print(f"✅ Best model saved to {best_path}")
