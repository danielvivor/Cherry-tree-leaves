import streamlit as st

from app_pages.page_summary import page_summary
from app_pages.page_leaf_visualizer import page_leaf_visualizer
from app_pages.page_mildew_detector import page_mildew_detector
from app_pages.page_project_hypotheses import page_project_hypotheses
from app_pages.page_ml_performance import page_ml_performance

# --------------------------------------------------
# Streamlit Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cherry Leaf Powdery Mildew Detector",
    page_icon="🍃",
    layout="wide",
)

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------

st.sidebar.title("🌿 Navigation Menu")

page = st.sidebar.radio(
    "Select Page",
    [
        "Summary",
        "Leaf Visualizer",
        "Powdery Mildew Detector",
        "Project Hypotheses",
        "ML Performance",
    ],
)

# --------------------------------------------------
# Routing
# --------------------------------------------------

if page == "Summary":
    page_summary()

elif page == "Leaf Visualizer":
    page_leaf_visualizer()

elif page == "Powdery Mildew Detector":
    page_mildew_detector()

elif page == "Project Hypotheses":
    page_project_hypotheses()

elif page == "ML Performance":
    page_ml_performance()