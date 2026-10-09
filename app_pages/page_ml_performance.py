import os
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st


def page_ml_performance():

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

    # --------------------------------------------------
    # Title
    # --------------------------------------------------

    st.title("📈 ML Performance")

    st.write(
        """
        This page evaluates the performance of the Convolutional Neural
        Network (CNN) that was developed to classify cherry leaves as
        either Healthy or Powdery Mildew.

        It addresses Business Requirement 2.
        """
    )

    # --------------------------------------------------
    # Overall Verdict
    # --------------------------------------------------

    st.success(
        """
        ✅ Final Model Verdict

        Business Requirement 2 required a machine learning model capable
        of identifying whether a cherry leaf is healthy or affected by
        powdery mildew.

        The final CNN model exceeded the project success criterion
        of 97% accuracy on unseen data and therefore successfully
        satisfies the business requirement.
        """
    )

    st.write("---")

    # --------------------------------------------------
    # Model Evolution
    # --------------------------------------------------

    st.header("Model Evolution & Hyperparameter Tuning")

    st.markdown(
        """
        ### Experiment 1
        - Baseline CNN architecture.
        - Initial model training established a performance baseline.
        - Results indicated opportunities for further optimisation.

        ### Experiment 2
        - Added image augmentation.
        - Improved model generalisation.
        - Reduced sensitivity to image orientation and variation.

        ### Experiment 3
        - Introduced Dropout layers.
        - Added Early Stopping monitoring.
        - Reduced overfitting and improved training stability.

        ### Final Model
        - Data augmentation enabled.
        - Dropout regularisation enabled.
        - Early Stopping enabled.
        - Selected based on validation performance.
        - Exceeded the project target accuracy of 97%.
        """
    )

    st.info(
        """
        Interpretation:

        Multiple model configurations were evaluated before selecting
        the final architecture.

        The final model was selected because it achieved the strongest
        balance between prediction accuracy and generalisation on
        previously unseen data.
        """
    )

    st.write("---")

    # --------------------------------------------------
    # Learning Curves
    # --------------------------------------------------

    st.header("Training Performance")

    history_path = os.path.join(
        outputs_dir,
        "training_history.pkl"
    )

    if os.path.exists(history_path):

        with open(history_path, "rb") as f:
            history = pickle.load(f)

        # Accuracy Plot

        df_accuracy = pd.DataFrame(
            {
                "Epoch": range(
                    1,
                    len(history["accuracy"]) + 1
                ),
                "Training Accuracy": history["accuracy"],
                "Validation Accuracy": history["val_accuracy"],
            }
        )

        fig_accuracy = px.line(
            df_accuracy,
            x="Epoch",
            y=[
                "Training Accuracy",
                "Validation Accuracy",
            ],
            title="Training vs Validation Accuracy",
        )

        st.plotly_chart(
            fig_accuracy,
            use_container_width=True
        )

        st.info(
            """
            Interpretation:

            Training and validation accuracy improve together
            throughout the training process.

            The close alignment between both curves suggests that
            the model generalises effectively and shows little
            evidence of overfitting.
            """
        )

        # Loss Plot

        df_loss = pd.DataFrame(
            {
                "Epoch": range(
                    1,
                    len(history["loss"]) + 1
                ),
                "Training Loss": history["loss"],
                "Validation Loss": history["val_loss"],
            }
        )

        fig_loss = px.line(
            df_loss,
            x="Epoch",
            y=[
                "Training Loss",
                "Validation Loss",
            ],
            title="Training vs Validation Loss",
        )

        st.plotly_chart(
            fig_loss,
            use_container_width=True
        )

        st.info(
            """
            Interpretation:

            Training and validation loss decrease steadily
            during optimisation.

            The absence of significant divergence indicates that
            the network learned relevant image features while
            maintaining stable performance on unseen validation
            samples.
            """
        )

    else:
        st.warning(
            "Training history file could not be found."
        )

    st.write("---")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    st.header("Confusion Matrix")

    confusion_matrix_path = os.path.join(
        outputs_dir,
        "confusion_matrix.png"
    )

    if os.path.exists(confusion_matrix_path):

        st.image(
            confusion_matrix_path,
            caption="Confusion Matrix for Test Dataset"
        )

        st.info(
            """
            Interpretation:

            The confusion matrix summarises prediction outcomes
            on the test dataset.

            From a business perspective, minimising false negatives
            is especially important because infected trees should
            not remain untreated.
            """
        )

    else:
        st.warning(
            "Confusion matrix image could not be found."
        )

    st.write("---")

    # --------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------

    st.header("Final Evaluation Metrics")

    evaluation_path = os.path.join(
        outputs_dir,
        "evaluation.pkl"
    )

    if os.path.exists(evaluation_path):

        with open(evaluation_path, "rb") as f:
            evaluation = pickle.load(f)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Test Loss",
                f"{evaluation['test_loss']:.4f}"
            )

        with col2:
            st.metric(
                "Test Accuracy",
                f"{evaluation['test_accuracy'] * 100:.2f}%"
            )

    else:
        st.warning(
            "Evaluation metrics file could not be found."
        )

    st.write("---")

    # --------------------------------------------------
    # Business Conclusion
    # --------------------------------------------------

    st.header("Business Conclusion")

    st.success(
        """
        The machine learning pipeline successfully learned the
        visual characteristics associated with powdery mildew.

        Evaluation on previously unseen test data demonstrates
        that the model can reliably distinguish healthy leaves
        from infected leaves.

        Therefore, Business Requirement 2 has been successfully
        achieved.
        """
    )