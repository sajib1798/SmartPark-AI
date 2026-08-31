from app.config import settings
from app.main import load_parking_slots


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

def test_project_name():

    assert settings.APP_NAME == "SmartPark AI"


def test_parking_config_exists():

    assert settings.PARKING_SLOTS_FILE.exists()


# ============================================================
# PARKING SLOT CONFIGURATION
# ============================================================

def test_parking_slots_loaded():

    data = load_parking_slots()

    assert "slots" in data

    assert isinstance(
        data["slots"],
        list,
    )

    assert len(data["slots"]) > 0


def test_each_slot_has_required_fields():

    data = load_parking_slots()

    for slot in data["slots"]:

        assert "id" in slot

        assert "name" in slot

        assert "points" in slot