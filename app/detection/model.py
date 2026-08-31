from pathlib import Path
from typing import Union

import torch
from ultralytics import YOLO

from app.config import settings


# ============================================================
# DEVICE SELECTION
# ============================================================

def get_device() -> Union[int, str]:
    """
    Select GPU when CUDA is available.
    Otherwise use CPU.
    """

    if torch.cuda.is_available():
        return 0

    return "cpu"


# ============================================================
# DEVICE DESCRIPTION
# ============================================================

def get_device_name() -> str:
    """
    Return human-readable training device name.
    """

    if torch.cuda.is_available():

        return torch.cuda.get_device_name(0)

    return "CPU"


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

def load_pretrained_model() -> YOLO:
    """
    Load pretrained YOLOv8n model.
    """

    model = YOLO(
        settings.PRETRAINED_MODEL
    )

    return model


# ============================================================
# LOAD CUSTOM TRAINED MODEL
# ============================================================

def load_trained_model(
    model_path: Path | None = None,
) -> YOLO:
    """
    Load the custom SmartPark model.
    """

    if model_path is None:
        model_path = settings.BEST_MODEL_FILE

    if not model_path.exists():

        raise FileNotFoundError(
            f"Trained model not found: "
            f"{model_path}"
        )

    return YOLO(
        str(model_path)
    )