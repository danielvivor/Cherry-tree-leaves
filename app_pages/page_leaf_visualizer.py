# Page 2: Visual Study (Averages, Differences)import streamlit as st
import os
import streamlit as st

def page_leaf_visualizer():

    PROJECT_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    OUTPUTS_DIR = os.path.join(
        PROJECT_DIR,
        "outputs",
        "v1"
    )

    st.title("📷 Leaf Visualizer (Visual Study)")

    st.write(
        "This page addresses Business Requirement 1 by examining the visual differences "
        "between healthy cherry leaves and leaves affected by powdery mildew."
    )

    if st.checkbox("View Average and Variability Plots", value=True):

        avg_healthy = os.path.join(
            OUTPUTS_DIR,
            "avg_var_healthy.png"
        )

        avg_mildew = os.path.join(
            OUTPUTS_DIR,
            "avg_var_powdery_mildew.png"
        )

        if os.path.exists(avg_healthy) and os.path.exists(avg_mildew):

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    avg_healthy,
                    caption="Healthy Leaf Average and Variability"
                )

            with col2:
                st.image(
                    avg_mildew,
                    caption="Powdery Mildew Average and Variability"
                )

        st.info(
            "Interpretation: Healthy leaf images display a more uniform colour distribution. "
            "Powdery mildew images contain lighter regions associated with fungal infection."
        )

    if st.checkbox("View Difference Between Averages"):

        avg_diff = os.path.join(
            OUTPUTS_DIR,
            "avg_diff.png"
        )

        if os.path.exists(avg_diff):

            st.image(
                avg_diff,
                caption="Difference Between Healthy and Mildew Average Images"
            )

            st.info(
                "Interpretation: The difference image highlights regions where infected leaves "
                "systematically differ from healthy leaves. This provides evidence that the two "
                "classes contain distinguishable visual patterns."
            )