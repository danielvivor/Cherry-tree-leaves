# Page 1: Project Summary & Business Requirements

import streamlit as st


def page_summary():
    st.title("🍃 Cherry Leaf Powdery Mildew Detection")
    st.subheader("Project Overview & Business Requirements")

    st.info(
        "**Powdery Mildew** is a destructive fungal disease affecting cherry trees. "
        "Manual crop inspection across thousands of leaves takes roughly 30 minutes per tree "
        "and is highly labour-intensive. This application delivers an automated, deep-learning "
        "image processing system capable of determining instantly whether a cherry leaf "
        "is healthy or infected."
    )

    st.header("📊 Dataset Content & Characteristics")

    st.markdown("""
    **Dataset Source**

    https://www.kaggle.com/codeinstitute/cherry-leaves

    **Classes**
    - Healthy
    - Powdery Mildew

    **Image Shape**
    - 256 x 256 pixels

    **Target Variable**
    - Leaf Health Status
    """)

    st.markdown("""
    ### 🎯 Business Requirements

    1. Conduct a visual study to differentiate healthy leaves from infected leaves.
    2. Deliver an accurate binary classification model capable of detecting powdery mildew.
    """)