from pathlib import Path

from app.config import settings

from app.detection.model import (
    get_device,
    get_device_name,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def test_pretrained_model_name():

    assert (
        settings.PRETRAINED_MODEL
        == "yolov8n.pt"
    )


def test_image_size():

    assert settings.IMAGE_SIZE > 0


def test_epochs():

    assert settings.EPOCHS > 0


def test_batch_size():

    assert settings.BATCH_SIZE > 0


# ============================================================
# DEVICE
# ============================================================

def test_training_device():

    device = get_device()

    assert (
        device == 0
        or device == "cpu"
    )


def test_device_name():

    device_name = get_device_name()

    assert isinstance(
        device_name,
        str,
    )

    assert len(device_name) > 0


# ============================================================
# DATASET YAML
# ============================================================

def test_dataset_yaml_path():

    assert isinstance(
        settings.DATA_YAML_FILE,
        Path,
    )