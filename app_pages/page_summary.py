# Page 1: Project Summary & Business Requirements

import streamlit as st


def page_summary():

    st.title("🍃 Cherry Leaf Powdery Mildew Detection")
    st.subheader("Project Overview & Business Requirements")

    st.info(
        "**Powdery Mildew** is a destructive fungal disease affecting cherry trees. "
        "Manual crop inspection across thousands of leaves takes roughly 30 minutes per tree "
        "and is highly labor-intensive. This application delivers an automated, deep-learning "
        "image processing system capable of determining instantly (<2 seconds) whether a cherry leaf "
        "is healthy or infected, creating a scalable diagnostic path for Farmy & Foods."
    )

    st.header("📊 Dataset Content & Characteristics")

    st.markdown("""
    **Dataset Source**

    https://www.kaggle.com/codeinstitute/cherry-leaves
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Images", "4,208")

    with col2:
        st.metric("Target Classes", "2")

    with col3:
        st.metric("Image Shape", "256 x 256")

    st.markdown("""
    ### 🏷️ Dataset Classes

    - **Healthy:** 2,104 images
    - **Powdery Mildew:** 2,104 images

    ### 📐 Dataset Split

    - **Training Set:** 2,944 images
    - **Validation Set:** 422 images
    - **Test Set:** 842 images
    """)

    st.markdown("""
    ### 🎯 Business Requirements

    1. Conduct a visual study to differentiate healthy and infected leaves.
    2. Build a machine learning model capable of predicting powdery mildew infection with high accuracy.
    """)