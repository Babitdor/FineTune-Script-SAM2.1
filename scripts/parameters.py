import json
from typing import Dict, Any


class TrainingParams:
    """Container for training parameters with type-safe updates and defaults."""

    # Parameters that should always be converted to float
    FLOAT_PARAMS = {
        "learning_rate",
        "score_weight",
        "weight_decay",
        "min_lr",
        
    }

    def __init__(self, **kwargs):
        """Initialize with default parameters, then update with any provided kwargs."""
        # Default parameter values
        self._params = {
            "output_dir": "models",
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "logging_strategy": "steps",
            "logging_dir": "logs",
            "logging_steps": 100,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "total_steps": 1000,
            "learning_rate": 3e-5,
            "min_lr": 1e-4,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "fp16": False,
            "bf16": False,
            "save_steps": 200,
            "eval_steps": 200,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "greater_is_better": True,
            "train_prompt_encoder": False,
            "train_mask_decoder": False,
            "train_image_encoder": False,
            "point_strategy": "gaussian",
            "num_pts": 5,
            "score_weight": 0.3,
            "use_lora": True,
            "lora_image_encoder": True,  # Apply LoRA to image encoder
            "lora_mask_decoder": True,  # Apply LoRA to mask decoder
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
        }

        # Apply any provided updates
        self.update(kwargs)

    def update(self, kwargs: Dict[str, Any]) -> None:
        """Safely update parameters with type checking for float values."""
        for key, value in kwargs.items():
            if key in self.FLOAT_PARAMS:
                try:
                    self._params[key] = float(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Invalid value for {key}: {value} (expected float)"
                    ) from e
            else:
                self._params[key] = value

    def __getattr__(self, key: str) -> Any:
        """Allow dot access to parameters."""
        if key in self._params:
            return self._params[key]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        """Handle setting both regular attributes and parameters."""
        if key == "_params":
            super().__setattr__(key, value)
        elif key in self.__dict__:
            super().__setattr__(key, value)
        else:
            self._params[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return parameters as a dictionary."""
        return self._params.copy()

    def __str__(self) -> str:
        """Return printed JSON representation of parameters."""
        return json.dumps(self._params, indent=2)
