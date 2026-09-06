from app.config import settings


# ============================================================
# CORE DIRECTORIES
# ============================================================

def test_core_directories_exist():

    assert settings.APP_DIR.exists()

    assert settings.DATA_DIR.exists()

    assert settings.MODELS_DIR.exists()

    assert settings.OUTPUT_DIR.exists()

    assert settings.CONFIG_DIR.exists()


# ============================================================
# CONFIGURATION
# ============================================================

def test_project_name():

    assert (
        settings.APP_NAME
        == "SmartPark AI"
    )


def test_image_size():

    assert (
        settings.IMAGE_SIZE
        > 0
    )


def test_confidence_threshold():

    assert (
        0.0
        <= settings.CONFIDENCE_THRESHOLD
        <= 1.0
    )


def test_vehicle_confidence_threshold():

    assert (
        0.0
        <= (
            settings
            .VEHICLE_CONFIDENCE_THRESHOLD
        )
        <= 1.0
    )


# ============================================================
# VEHICLE CLASSES
# ============================================================

def test_vehicle_classes():

    expected = {
        "car",
        "motorcycle",
        "bus",
        "truck",
    }

    assert (
        settings.VEHICLE_CLASS_NAMES
        == expected
    )


# ============================================================
# PARKING CLASSES
# ============================================================

def test_parking_class_names():

    assert (
        settings.EMPTY_CLASS_NAME
        == "space-empty"
    )

    assert (
        settings.OCCUPIED_CLASS_NAME
        == "space-occupied"
    )