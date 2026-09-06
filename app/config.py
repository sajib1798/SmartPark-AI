from pathlib import Path


class Settings:
    """
    Central configuration settings for SmartPark AI.
    """

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    APP_NAME = "SmartPark AI"
    VERSION = "1.0.0"
    DEBUG = True

    # ========================================================
    # PROJECT ROOT
    # ========================================================

    BASE_DIR = Path(__file__).resolve().parent.parent

    # ========================================================
    # MAIN DIRECTORIES
    # ========================================================

    APP_DIR = BASE_DIR / "app"
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    OUTPUT_DIR = BASE_DIR / "outputs"
    CONFIG_DIR = BASE_DIR / "config"

    # ========================================================
    # DATASET DIRECTORIES
    # ========================================================

    DATASET_DIR = DATA_DIR / "dataset"

    TRAIN_DIR = DATASET_DIR / "train"
    VALID_DIR = DATASET_DIR / "valid"
    TEST_DIR = DATASET_DIR / "test"

    TRAIN_IMAGES_DIR = TRAIN_DIR / "images"
    TRAIN_LABELS_DIR = TRAIN_DIR / "labels"

    VALID_IMAGES_DIR = VALID_DIR / "images"
    VALID_LABELS_DIR = VALID_DIR / "labels"

    TEST_IMAGES_DIR = TEST_DIR / "images"
    TEST_LABELS_DIR = TEST_DIR / "labels"

    DATA_YAML_FILE = DATASET_DIR / "data.yaml"

    # ========================================================
    # PARKING SLOT CONFIGURATION
    # ========================================================

    PARKING_SLOTS_FILE = (
        CONFIG_DIR / "parking_slots.json"
    )

    # ========================================================
    # PHASE 1 - DATASET
    # ========================================================

    DATASET_REPORT_FILE = (
        OUTPUT_DIR / "dataset_report.json"
    )

    # ========================================================
    # PHASE 2 - YOLO TRAINING
    # ========================================================

    PRETRAINED_MODEL = "yolov8n.pt"

    IMAGE_SIZE = 640

    EPOCHS = 50

    BATCH_SIZE = 8

    WORKERS = 2

    CONFIDENCE_THRESHOLD = 0.25

    IOU_THRESHOLD = 0.45

    TRAINING_OUTPUT_DIR = (
        OUTPUT_DIR / "training"
    )

    TRAINING_RUN_NAME = (
        "smartpark_yolov8n"
    )

    BEST_MODEL_FILE = (
        TRAINING_OUTPUT_DIR
        / TRAINING_RUN_NAME
        / "weights"
        / "best.pt"
    )

    LAST_MODEL_FILE = (
        TRAINING_OUTPUT_DIR
        / TRAINING_RUN_NAME
        / "weights"
        / "last.pt"
    )

    # ========================================================
    # PHASE 3 - INFERENCE
    # ========================================================

    PREDICTION_OUTPUT_DIR = (
        OUTPUT_DIR / "predictions"
    )

    VIDEO_OUTPUT_DIR = (
        OUTPUT_DIR / "videos"
    )

    EMPTY_CLASS_NAME = "space-empty"

    OCCUPIED_CLASS_NAME = "space-occupied"

    VIDEO_CODEC = "mp4v"

    # ========================================================
    # PHASE 4 - VEHICLE DETECTION
    # ========================================================

    # IMPORTANT:
    # Put pretrained YOLOv8 model here:
    #
    # E:/SmartPark_AI/models/yolov8n.pt

    VEHICLE_MODEL = (
        MODELS_DIR / "yolov8n.pt"
    )

    VEHICLE_CONFIDENCE_THRESHOLD = 0.30

    VEHICLE_IOU_THRESHOLD = 0.45

    VEHICLE_CLASS_NAMES = {
        "car",
        "motorcycle",
        "bus",
        "truck",
    }

    COMBINED_OUTPUT_DIR = (
        OUTPUT_DIR / "combined_predictions"
    )

    # ========================================================
    # PHASE 5 - STREAMLIT UI
    # ========================================================

    UI_TITLE = "SmartPark AI"

    UI_SUBTITLE = (
        "AI-Powered Parking Occupancy "
        "& Vehicle Detection System"
    )

    EVALUATION_METRICS_FILE = (
        OUTPUT_DIR / "evaluation_metrics.json"
    )

    TEMP_DIR = (
        OUTPUT_DIR / "temp"
    )

    # ========================================================
    # SUPPORTED IMAGE EXTENSIONS
    # ========================================================

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    # ========================================================
    # SUPPORTED VIDEO EXTENSIONS
    # ========================================================

    VIDEO_EXTENSIONS = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }


# ============================================================
# GLOBAL SETTINGS OBJECT
# ============================================================

settings = Settings()