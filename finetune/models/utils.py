from typing import Dict, Literal

from finetune.trainer import Trainer


SUPPORTED_MODELS: Dict[str, Dict[str, Trainer]] = {}


def register(model_name: str, training_type: Literal["lora", "sft", "kd"], trainer_cls: Trainer):
    """Register a model and its associated functions for a specific training type.

    Args:
        model_name (str): Name of the model to register (e.g. "cogvideox-5b")
        training_type (Literal["lora", "sft"]): Type of training - either "lora" or "sft"
        trainer_cls (Trainer): Trainer class to register.
    """

    # Check if model_name and training_type exists in SUPPORTED_MODELS
    if model_name not in SUPPORTED_MODELS:
        SUPPORTED_MODELS[model_name] = {}
    else:
        if training_type in SUPPORTED_MODELS[model_name]:
            raise ValueError(f"Training type {training_type} already exists for model {model_name}")

    SUPPORTED_MODELS[model_name][training_type] = trainer_cls


def show_supported_models():
    """Print all currently supported models and their training types."""

    print("\nSupported Models:")
    print("================")

    for model_name, training_types in SUPPORTED_MODELS.items():
        print(f"\n{model_name}")
        print("-" * len(model_name))
        for training_type in training_types:
            print(f"  • {training_type}")


def get_model_cls(model_type: str, training_type: Literal["lora", "sft"]) -> Trainer:
    """Get the trainer class for a specific model and training type."""
    if model_type not in SUPPORTED_MODELS:
        print(f"\nModel '{model_type}' is not supported.")
        print("\nSupported models are:")
        for supported_model in SUPPORTED_MODELS:
            print(f"  • {supported_model}")
        raise ValueError(f"Model '{model_type}' is not supported")

    if training_type not in SUPPORTED_MODELS[model_type]:
        print(f"\nTraining type '{training_type}' is not supported for model '{model_type}'.")
        print(f"\nSupported training types for '{model_type}' are:")
        for supported_type in SUPPORTED_MODELS[model_type]:
            print(f"  • {supported_type}")
        raise ValueError(
            f"Training type '{training_type}' is not supported for model '{model_type}'"
        )

    return SUPPORTED_MODELS[model_type][training_type]
