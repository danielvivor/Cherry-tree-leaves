import os
import streamlit as st
from PIL import Image

from src.data_management import load_model_and_classes
from src.machine_learning import predict_leaf

def page_mildew_detector():

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    outputs_dir = os.path.join(
        project_dir,
        "outputs",
        "v1"
    )

    model_path = os.path.join(
        outputs_dir,
        "powdery_mildew_detector_model.h5"
    )

    class_indices_path = os.path.join(
        outputs_dir,
        "class_indices.pkl"
    )

    # --------------------------------------------------
    # Page Content
    # --------------------------------------------------

    st.title("🔬 Powdery Mildew Detector")

    st.write(
        """
        Upload one or more cherry leaf images to predict whether the
        leaf is Healthy or affected by Powdery Mildew.

        This page addresses Business Requirement 2.
        """
    )

    if not os.path.exists(class_indices_path):

        st.warning(
            "Required model files were not found in outputs/v1."
        )

        return

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    model, map_labels = load_model_and_classes(
        model_path,
        class_indices_path,
    )

    # --------------------------------------------------
    # File Upload
    # --------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload Cherry Leaf Images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    st.write("---")

    results = []

    cols = st.columns(
        min(len(uploaded_files), 3)
    )

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    for idx, file in enumerate(uploaded_files):

        image = Image.open(file)

        prediction = predict_leaf(
            image,
            model,
            map_labels,
        )

        with cols[idx % 3]:

            st.image(
                image,
                caption=file.name,
            )

            if "Mildew" in prediction["Diagnostic"]:

                st.error(
                    f"""
                    {prediction['Diagnostic']}

                    Confidence:
                    {prediction['Confidence (%)']}%
                    """
                )

            else:

                st.success(
                    f"""
                    {prediction['Diagnostic']}

                    Confidence:
                    {prediction['Confidence (%)']}%
                    """
                )

        results.append(
            {
                "Image Name": file.name,
                "Diagnostic": prediction["Diagnostic"],
                "Confidence (%)": prediction["Confidence (%)"],
                "Raw Probability": prediction["Raw Probability"],
            }
        )

    # --------------------------------------------------
    # Prediction Summary
    # --------------------------------------------------

    st.write("---")
    st.subheader("Prediction Summary")

    for result in results:

        st.markdown(
            f"""
            **Image Name:** {result['Image Name']}

            **Diagnostic:** {result['Diagnostic']}

            **Confidence:** {result['Confidence (%)']}%

            **Raw Probability:** {result['Raw Probability']}
            """
        )

        st.write("---")

    # --------------------------------------------------
    # CSV Download
    # --------------------------------------------------

    csv_lines = [
        "Image Name,Diagnostic,Confidence (%),Raw Probability"
    ]

    for result in results:

        csv_lines.append(
            f"{result['Image Name']},"
            f"{result['Diagnostic']},"
            f"{result['Confidence (%)']},"
            f"{result['Raw Probability']}"
        )

    csv_data = "\n".join(csv_lines).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv_data,
        file_name="powdery_mildew_predictions.csv",
        mime="text/csv",
    )

    # --------------------------------------------------
    # Interpretation
    # --------------------------------------------------

    st.info(
        """
        Interpretation:

        The detector predicts whether a cherry leaf is healthy or
        infected using the trained CNN model.

        Higher confidence values indicate stronger model certainty
        regarding the classification outcome.
        """
    )