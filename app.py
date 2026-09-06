# Main Streamlit entry point with sidebar navigation

import os
import sys

# Resolve Windows DLL loading for TensorFlow
if sys.platform == "win32":
    os.add_dll_directory(r"C:\Windows\System32")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from src.data_management import load_model_and_classes, load_pkl_data
from src.machine_learning import predict_leaf

# Dashboard Page Configuration
st.set_page_config(
    page_title="Cherry Leaf Mildew Detector",
    page_icon="🍃",
    layout="wide"
)

# Define Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'outputs', 'v1')
MODEL_PATH = os.path.join(OUTPUTS_DIR, 'powdery_mildew_detector_model.h5')
CLASS_INDICES_PATH = os.path.join(OUTPUTS_DIR, 'class_indices.pkl')

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Option:",
    [
        "Summary",
        "Leaf Visualizer",
        "Powdery Mildew Detector",
        "Project Hypotheses",
        "ML Performance"
    ]
)

# -------------------------------------------------------------------
# Page 1: Summary
# -------------------------------------------------------------------
if page == "Summary":
    st.title("🍃 Cherry Leaf Powdery Mildew Detection")
    st.subheader("Project Overview & Business Requirements")

    st.info(
        "**Powdery Mildew** is a fungal disease affecting cherry trees. "
        "Manual inspection across thousands of leaves is labor-intensive and error-prone. "
        "This application delivers an automated image processing system capable of determining "
        "instantly whether a cherry leaf is healthy or infected."
    )

    st.markdown("""
    ### Business Requirements:
    1. **Requirement 1:** Conduct a visual study to differentiate healthy leaves from infected leaves using average images, standard deviation plots, and image montages.
    2. **Requirement 2:** Deliver an accurate binary classification model predicting with at least **97% accuracy** whether a cherry leaf contains powdery mildew.
    """)

# -------------------------------------------------------------------
# Page 2: Leaf Visualizer
# -------------------------------------------------------------------
elif page == "Leaf Visualizer":
    st.title("📊 Leaf Visualizer (Visual Study)")

    if st.checkbox("View Average and Variability Plots"):
        avg_healthy = os.path.join(OUTPUTS_DIR, 'avg_var_healthy.png')
        avg_mildew = os.path.join(OUTPUTS_DIR, 'avg_var_powdery_mildew.png')

        if os.path.exists(avg_healthy) and os.path.exists(avg_mildew):
            col1, col2 = st.columns(2)
            with col1:
                st.image(avg_healthy, caption="Healthy Leaf: Average & Std Dev")
            with col2:
                st.image(avg_mildew, caption="Powdery Mildew Leaf: Average & Std Dev")

    if st.checkbox("View Difference Between Averages"):
        avg_diff = os.path.join(OUTPUTS_DIR, 'avg_diff.png')
        if os.path.exists(avg_diff):
            st.image(avg_diff, caption="Visual Difference Between Healthy and Infected Average Images")

# -------------------------------------------------------------------
# Page 3: Powdery Mildew Detector
# -------------------------------------------------------------------
elif page == "Powdery Mildew Detector":
    st.title("🔬 Live Mildew Detector")
    st.write("Upload cherry leaf images to predict infection status in real time.")

    # Load Model Artifacts
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_INDICES_PATH):
        model, map_labels = load_model_and_classes(MODEL_PATH, CLASS_INDICES_PATH)
        
        uploaded_files = st.file_uploader(
            "Choose leaf image(s)...", 
            type=['png', 'jpg', 'jpeg'], 
            accept_multiple_files=True
        )

        if uploaded_files:
            results = []
            cols = st.columns(min(len(uploaded_files), 3))

            for idx, file in enumerate(uploaded_files):
                img_pil = Image.open(file)
                prediction = predict_leaf(img_pil, model, map_labels)

                # Render preview
                with cols[idx % 3]:
                    st.image(img_pil, caption=file.name, use_container_width=True)
                    if "Mildew" in prediction['Diagnostic']:
                        st.error(f"**{prediction['Diagnostic']}**\n\nConfidence: {prediction['Confidence (%)']}%")
                    else:
                        st.success(f"**{prediction['Diagnostic']}**\n\nConfidence: {prediction['Confidence (%)']}%")

                results.append({
                    "Image Name": file.name,
                    "Diagnostic": prediction['Diagnostic'],
                    "Confidence (%)": prediction['Confidence (%)'],
                    "Raw Probability": prediction['Raw Probability']
                })

            st.markdown("### Prediction Summary Table")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results)

            # Download CSV Report
            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Diagnostic CSV Report",
                data=csv_data,
                file_name="powdery_mildew_predictions.csv",
                mime="text/csv"
            )
    else:
        st.warning("Model or class index mapping not found in `outputs/v1/`. Please train the model in Notebook 3 first.")

# -------------------------------------------------------------------
# Page 4: Project Hypotheses
# -------------------------------------------------------------------
elif page == "Project Hypotheses":
    st.title("💡 Project Hypotheses & Validation")

    st.markdown("""
    * **Hypothesis 1:** Leaves infected with powdery mildew exhibit white, powdery circular patches across their surfaces that can be differentiated using pixel color distribution analysis.
      * **Validation:** Validated via image average and standard deviation analysis in Notebook 2.
    * **Hypothesis 2:** A Convolutional Neural Network (CNN) can achieve >97% accuracy on unseen test data.
      * **Validation:** Confirmed in Notebook 3 during model evaluation on the test set.
    """)

# -------------------------------------------------------------------
# Page 5: ML Performance
# -------------------------------------------------------------------
elif page == "ML Performance":
    st.title("📈 Model Performance Metrics")

    # Display History Plot
    history_plot = os.path.join(OUTPUTS_DIR, 'model_training_history.png')
    if os.path.exists(history_plot):
        st.subheader("Training History (Accuracy & Loss)")
        st.image(history_plot)

    # Display Confusion Matrix
    cm_plot = os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')
    if os.path.exists(cm_plot):
        st.subheader("Confusion Matrix")
        st.image(cm_plot)

    # Display Evaluation Metrics
    eval_path = os.path.join(OUTPUTS_DIR, 'evaluation.pkl')
    if os.path.exists(eval_path):
        eval_data = load_pkl_data(eval_path)
        col1, col2 = st.columns(2)
        col1.metric("Test Loss", f"{eval_data['test_loss']:.4f}")
        col2.metric("Test Accuracy", f"{eval_data['test_accuracy'] * 100:.2f}%")