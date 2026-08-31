# ============================================================
# IMPORT PATH FIX
# ============================================================

import sys
from pathlib import Path


# Current file:
# E:\SmartPark_AI\app\ui\streamlit_app.py

# Go back to:
# E:\SmartPark_AI
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add project root to Python import path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from app.config import settings


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartPark AI",
    page_icon="🚗",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("🚗 SmartPark AI")

st.subheader("AI-Powered Parking Occupancy Detection")


# ============================================================
# BASIC TEST
# ============================================================

st.success("SmartPark AI application started successfully!")


# ============================================================
# CONFIG TEST
# ============================================================

st.write("### Application Configuration")

st.write("App Name:", settings.APP_NAME)
st.write("Version:", settings.VERSION)


# ============================================================
# MAIN APPLICATION
# ============================================================

st.write("---")

st.write(
    """
    SmartPark AI will detect vehicles in parking areas
    and determine whether parking spaces are occupied or vacant.
    """
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a parking image",
    type=["jpg", "jpeg", "png"],
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Uploaded Parking Image",
        use_container_width=True,
    )

    st.info(
        "Object detection will be added in the next phase."
    )