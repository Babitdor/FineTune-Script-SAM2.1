import json


class TrainingParams:
    def __init__(self, **kwargs):
        self.params = {
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
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "fp16": False,
            "bf16": False,
            "save_steps": 200,
            "eval_steps": 200,
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "greater_is_better": True,
            "train_prompt_encoder": True,
            "train_mask_decoder": True,
            "bce_weight": 0.2,
            "dice_weight": 0.5,
            "score_weight": 0.3,
        }
        self.params.update(kwargs)

    def __getattr__(self, key):
        return self.params.get(key)

    def __setattr__(self, key, value):
        if key == "params":
            super().__setattr__(key, value)
        else:
            self.params[key] = value

    def to_dict(self):
        return self.params

    def __str__(self):
        return json.dumps(self.params, indent=2)
