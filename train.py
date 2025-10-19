from torch.utils.data import DataLoader
from scripts.dataloader import SENSATIONDataset
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from scripts.parameters import TrainingParams
from peft import PeftModel
from hydra import initialize_config_dir
from numpy._core.multiarray import scalar
from hydra.core.global_hydra import GlobalHydra
import os
import torch
import sys
import yaml
import glob
from scripts.trainer import Trainer
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM2 model")

    # Model name to config file mapping
    model_config_map = {
        "sam2.1_hiera_large.pt": "sam2.1_hiera_l.yaml",
        "sam2.1_hiera_base_plus.pt": "sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_small.pt": "sam2.1_hiera_s.yaml",
        "sam2.1_hiera_tiny.pt": "sam2.1_hiera_t.yaml",
    }

    # Only allow overriding these specific paths
    parser.add_argument(
        "--initial-config-file",
        type=str,
        default="configs/initialize_config.yaml",
        help="Path to initialize config file (default: configs/initialize_config.yaml)",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default="sam2/pretrained_models/",
        help="Override PRETRAINED_MODEL_PATH from config (default: sam2/configs/sam2.1)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sam2.1_hiera_large.pt",
        choices=list(model_config_map.keys()),
        help="Override MODEL_NAME from config",
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="sam2/configs/sam2.1/",
        help="Path to model configs folder (default: sam2/configs/sam2.1)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="models/checkpoint",
        help="Override CHECKPOINT_PATH from config",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="To visualize point selection"
    )

    args = parser.parse_args()

    # Automatically set the config name based on model name
    args.model_config_name = model_config_map[args.model_name]

    return args


def load_training_params(config):
    """Convert config dictionary to TrainingParams object with fallback to defaults"""
    training_config = config.get("Training_parameters", {})

    if not training_config:
        print("Using all default training parameters")
    else:
        print(f"Loading {len(training_config)} parameter(s) from config")
        print("(Missing parameters will use defaults)")

    return TrainingParams(**training_config)


def main():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    args = parse_args()

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Load config file
    with open(args.initial_config_file, "r") as f:
        config = yaml.safe_load(f)

    # Override specific paths if provided in command line
    if args.pretrained_model_path is not None:
        config["PRETRAINED_MODEL_PATH"] = args.pretrained_model_path

    if args.model_name is not None:
        config["MODEL_NAME"] = args.model_name
        print(f"Pre-Trained Model Loaded : {config['MODEL_NAME']}")

    if args.checkpoint_path is not None:
        config["CHECKPOINT_PATH"] = args.checkpoint_path

    if args.model_config_path is not None:
        config["CONFIG_PATH"] = args.model_config_path

    if args.model_config_name is not None:
        config["CONFIG_NAME"] = args.model_config_name
        print(f"Pre-Trained Config Loaded : {config['CONFIG_NAME']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training parameters from config
    training_args = load_training_params(config)

    # Initialize datasets
    train_dataset = SENSATIONDataset(root_dir=config["DATA_PATH"], split="training")
    val_dataset = SENSATIONDataset(root_dir=config["DATA_PATH"], split="validation")

    model_config_dir = os.path.abspath(args.model_config_path)
    model_config_file = os.path.join(model_config_dir, args.model_config_name)

    # Verify config exists
    if not os.path.exists(model_config_file):
        raise FileNotFoundError(
            f"Model config not found at: {model_config_file}\n"
            f"Config directory contents: {os.listdir(model_config_dir)}"
        )
    with initialize_config_dir(version_base="1.2", config_dir=model_config_dir):
        # Initialize model
        sam_model = build_sam2(
            config_file=args.model_config_name.replace(".yaml", ""),
            ckpt_path=os.path.abspath(
                os.path.join(config["PRETRAINED_MODEL_PATH"], config["MODEL_NAME"])
            ),
            device=device,  # type: ignore
        )
    predictor = SAM2ImagePredictor(sam_model)

    ## Commented Code for Checking of named_modules of the componentes
    # Mask Decoder
    # for name, module in predictor.model.sam_mask_decoder.named_modules():
    #     print(name, type(module))

    # # Prompt Decoder
    # for name, module in predictor.model.sam_prompt_encoder.named_modules():
    #     print(name, type(module))

    # Image Encoder
    # for name, module in predictor.model.image_encoder.named_modules():
    #     print(name, type(module))

    trainer = Trainer(
        params=training_args,
        predictor=predictor,
        train_data=train_dataset,
        val_data=val_dataset,
        model_name=args.model_config_name,
        device=device,  # type: ignore
        logging=True,
        tensorboard_logging=True,
        visualize=args.visualize,
    )

    if args.resume:
        checkpoint_files = sorted(
            glob.glob(os.path.join(config["CHECKPOINT_PATH"], "checkpoint_step_*.pt")),
            key=os.path.getmtime,
            reverse=True,
        )
        if checkpoint_files:
            checkpoint_path = checkpoint_files[0]
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            trainer.predictor.model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.global_step = checkpoint.get("step", 0)
            trainer.best_metric = checkpoint.get("best_metric", trainer.best_metric)
            trainer.params = TrainingParams(**checkpoint.get("params", {}))
            print(f"Resumed from checkpoint at step {trainer.global_step}")
            print(f"Current best IoU achieved: {trainer.best_metric:.4f}")
        else:
            print("No checkpoint found, starting fresh.")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving checkpoint...")
        trainer.save_checkpoint(trainer.global_step)
        print("Checkpoint saved. Exiting.")


if __name__ == "__main__":
    main()
