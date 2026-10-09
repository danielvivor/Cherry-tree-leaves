import os
import streamlit as st


def page_leaf_visualizer():

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    outputs_dir = os.path.join(
        project_dir,
        "outputs",
        "v1"
    )

    st.title("📷 Leaf Visualizer")
    st.subheader("Business Requirement 1")

    st.write(
        """
        This page investigates the visual characteristics of healthy
        cherry leaves and leaves affected by powdery mildew.

        The objective is to determine whether objective visual
        differences exist between the two classes.
        """
    )

    st.write("---")

    # ==================================================
    # Average and Variability Study
    # ==================================================

    st.header("Average and Variability Analysis")

    avg_healthy = os.path.join(
        outputs_dir,
        "avg_var_healthy.png"
    )

    avg_mildew = os.path.join(
        outputs_dir,
        "avg_var_powdery_mildew.png"
    )

    if os.path.exists(avg_healthy) and os.path.exists(avg_mildew):

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                avg_healthy,
                caption="Healthy Leaves: Average and Variability"
            )

        with col2:
            st.image(
                avg_mildew,
                caption="Powdery Mildew Leaves: Average and Variability"
            )

        st.info(
            """
            Interpretation:

            Healthy leaves exhibit a more uniform colour distribution
            and texture pattern.

            Powdery mildew leaves contain lighter regions associated
            with fungal infection. The variability plots indicate that
            infected leaves retain recurring visual characteristics
            despite natural variation between samples.
            """
        )

    st.write("---")

    # ==================================================
    # Difference Image
    # ==================================================

    st.header("Difference Between Average Images")

    avg_diff = os.path.join(
        outputs_dir,
        "avg_diff.png"
    )

    if os.path.exists(avg_diff):

        st.image(
            avg_diff,
            caption="Difference Between Healthy and Powdery Mildew Averages"
        )

        st.info(
            """
            Interpretation:

            The difference image highlights the regions where healthy
            and infected leaves differ most strongly.

            The visible contrast supports the hypothesis that powdery
            mildew produces consistent image features that can be
            learned by a machine learning model.
            """
        )

    st.write("---")

    # ==================================================
    # Image Montages
    # ==================================================

    st.header("Leaf Sample Montages")

    healthy_montage = os.path.join(
        outputs_dir,
        "montage_healthy.png"
    )

    mildew_montage = os.path.join(
        outputs_dir,
        "montage_powdery_mildew.png"
    )

    if os.path.exists(healthy_montage) and os.path.exists(mildew_montage):

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                healthy_montage,
                caption="Healthy Leaf Montage"
            )

        with col2:
            st.image(
                mildew_montage,
                caption="Powdery Mildew Montage"
            )

        st.info(
            """
            Interpretation:

            The montage visualises multiple sample leaves from each
            class.

            Healthy leaves generally display a consistent green
            appearance, while powdery mildew samples show recurring
            pale fungal markings that differentiate them from healthy
            foliage.
            """
        )

    st.write("---")

    # ==================================================
    # Image Dimensions
    # ==================================================

    st.header("Image Dimension Analysis")

    image_dimensions = os.path.join(
        outputs_dir,
        "image_dimensions.png"
    )

    if os.path.exists(image_dimensions):

        st.image(
            image_dimensions,
            caption="Image Dimension Distribution"
        )

        st.info(
            """
            Interpretation:

            The image dimension analysis demonstrates the consistency
            of image sizes across the dataset.

            Consistent dimensions simplify preprocessing and ensure
            that image resizing introduces minimal distortion during
            model training.
            """
        )

    st.write("---")

    # ==================================================
    # Pixel Intensity Histogram
    # ==================================================

    st.header("Pixel Intensity Distribution")

    histogram = os.path.join(
        outputs_dir,
        "pixel_intensity_histogram.png"
    )

    if os.path.exists(histogram):

        st.image(
            histogram,
            caption="Pixel Intensity Histogram"
        )

        st.info(
            """
            Interpretation:

            The histogram compares colour intensity distributions
            across the dataset.

            Differences in pixel intensity provide quantitative
            evidence that healthy and infected leaves occupy different
            regions of colour space, supporting visual discrimination.
            """
        )

    st.write("---")

    # ==================================================
    # Class Distribution
    # ==================================================

    st.header("Dataset Class Distribution")

    class_dist = os.path.join(
        outputs_dir,
        "class_distribution.png"
    )

    if os.path.exists(class_dist):

        st.image(
            class_dist,
            caption="Class Distribution"
        )

        st.info(
            """
            Interpretation:

            The dataset is balanced across healthy and powdery mildew
            classes.

            Balanced class representation reduces model bias and helps
            ensure that prediction performance reflects genuine
            learning rather than favouring a dominant class.
            """
        )

    st.write("---")

    st.success(
        """
        Conclusion:

        Healthy and powdery mildew leaves show observable visual
        differences in colour, texture and pixel characteristics.

        The visual study therefore provides evidence supporting
        Business Requirement 1 and justifies the use of machine
        learning for automated classification.
        """
    )