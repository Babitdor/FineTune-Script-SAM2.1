from torch.utils.data import DataLoader
from scripts.dataloader import SENSATIONDataset
from sam2 import build_sam, sam2_image_predictor
from scripts.transforms import get_training_augmentation
from scripts.parameters import TrainingParams
import os
import torch
import sys
import yaml
import glob
from scripts.trainer import Trainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
with open("sam2/configs/fine_tune_sidewalk/initialize_config.yaml", "r") as f:
    initialize = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Training Arguments
training_args = TrainingParams(
    output_dir="models",
    logging_dir="logs",
    logging_steps=100,
    gradient_accumulation_steps=8,
    total_steps=5000,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    greater_is_better=True,
    train_prompt_encoder=True,
    train_mask_decoder=True,
    bce_weight=0.2,
    dice_weight=0.5,
    score_weight=0.3,
)
# Initialize datasets

train_dataset = SENSATIONDataset(manifest_path=initialize["TRAIN_MANIFEST"])
val_dataset = SENSATIONDataset(manifest_path=initialize["VAL_MANIFEST"])

# Initialize data loaders
train_data = DataLoader(
    train_dataset,
    shuffle=True,
)
val_data = DataLoader(
    val_dataset,
    shuffle=True,
)
# Initialize model
sam_model = build_sam.build_sam2(
    config_file=os.path.join(initialize["CONFIG_PATH"], initialize["CONFIG_NAME"]),
    ckpt_path=os.path.join(
        initialize["PRETRAINED_MODEL_PATH"], initialize["MODEL_NAME"]
    ),
    device=device,
)
predictor = sam2_image_predictor.SAM2ImagePredictor(sam_model)

# Create and run trainer
trainer = Trainer(
    params=training_args,
    predictor=predictor,
    train_data=train_data,
    val_data=val_data,
    device=device,
    logging=True,
    tensorboard_logging=True,
)

# Loading checkpoint model if available, and resume from checkpoint!
checkpoint_files = sorted(
    glob.glob(os.path.join(initialize["CHECKPOINT_PATH"], "checkpoint_step_*.pt")),
    key=os.path.getmtime,
    reverse=True,
)
if checkpoint_files:
    checkpoint_path = checkpoint_files[0]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trainer.predictor.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.global_step = checkpoint.get("step", 0)
    trainer.best_metric = checkpoint.get("best_metric", trainer.best_metric)
    trainer.params = TrainingParams(**checkpoint.get("params", {}))
    print(f"Resumed from checkpoint at step {trainer.global_step}")
else:
    print("No checkpoint found, starting fresh.")

try:
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted by user. Saving checkpoint...")
    trainer.save_checkpoint(trainer.global_step)
    print("Checkpoint saved. Exiting.")
