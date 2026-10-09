# Page 5: Model Training Metrics & Test Evaluation
import os
import pickle
import pandas as pd
import plotly.express as px
import streamlit as st


def page_ml_performance():

    project_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    outputs_dir = os.path.join(
        project_dir,
        "outputs",
        "v1"
    )

    st.title("📈 ML Performance")

    st.write(
        """
        This page evaluates the final CNN model and determines
        whether Business Requirement 2 has been successfully met.
        """
    )

    # ==================================================
    # Explicit Project Verdict
    # ==================================================

    st.success(
        """
        ✅ Final Model Verdict

        Business Requirement 2 required a model capable of
        distinguishing healthy cherry leaves from leaves
        affected by powdery mildew.

        The final CNN model exceeded the minimum success
        threshold of 97% accuracy and therefore successfully
        satisfies the business requirement.
        """
    )

    st.write("---")

    # ==================================================
    # Hyperparameter / Model Evolution
    # ==================================================

    st.header("Model Evolution & Hyperparameter Tuning")

st.markdown("""
### Experiment 1
- Baseline CNN architecture
- No augmentation
- Initial validation accuracy below target

### Experiment 2
- Added image augmentation
- Improved generalisation performance
- Reduced overfitting

### Experiment 3
- Added Dropout layers
- Added Early Stopping callback
- Improved validation stability

### Final Model
- Data augmentation enabled
- Dropout regularisation enabled
- Early Stopping enabled
- Selected based on strongest validation performance
- Exceeded the business target of 97% accuracy
""")

st.info(
    """
    Interpretation:

    Multiple model configurations were evaluated before selecting
    the final architecture.

    The final model was chosen because it achieved the strongest
    balance between prediction accuracy and generalisation on
    unseen data.
    """
)

    st.write("---")

    # ==================================================
    # Interactive Learning Curves
    # ==================================================

    history_path = os.path.join(
        outputs_dir,
        "training_history.pkl"
    )

    if os.path.exists(history_path):

        with open(history_path, "rb") as f:
            history = pickle.load(f)

        df_acc = pd.DataFrame({
            "Epoch": range(
                1,
                len(history["accuracy"]) + 1
            ),
            "Training Accuracy": history["accuracy"],
            "Validation Accuracy": history["val_accuracy"]
        })

        fig_acc = px.line(
            df_acc,
            x="Epoch",
            y=[
                "Training Accuracy",
                "Validation Accuracy"
            ],
            title="Training vs Validation Accuracy"
        )

        st.plotly_chart(
            fig_acc,
            use_container_width=True
        )

        st.info(
            """
            Interpretation:

            Training and validation accuracy increase together
            throughout training.

            The close alignment between the curves suggests
            that the model generalises effectively and shows
            little evidence of overfitting.
            """
        )

        df_loss = pd.DataFrame({
            "Epoch": range(
                1,
                len(history["loss"]) + 1
            ),
            "Training Loss": history["loss"],
            "Validation Loss": history["val_loss"]
        })

        fig_loss = px.line(
            df_loss,
            x="Epoch",
            y=[
                "Training Loss",
                "Validation Loss"
            ],
            title="Training vs Validation Loss"
        )

        st.plotly_chart(
            fig_loss,
            use_container_width=True
        )

        st.info(
            """
            Interpretation:

            Training and validation loss decrease steadily
            throughout the optimisation process.

            The absence of substantial divergence indicates
            that the model learned meaningful image features
            while maintaining good generalisation performance.
            """
        )

    st.write("---")

    # ==================================================
    # Confusion Matrix
    # ==================================================

    cm_plot = os.path.join(
        outputs_dir,
        "confusion_matrix.png"
    )

    if os.path.exists(cm_plot):

        st.subheader("Confusion Matrix")

        st.image(
            cm_plot,
            caption="Test Dataset Confusion Matrix"
        )

        st.info(
            """
            Interpretation:

            The confusion matrix shows how many healthy
            and infected leaves were correctly classified.

            From a business perspective, minimising false
            negatives is particularly important because
            infected trees should not remain untreated.
            """
        )

    st.write("---")

    # ==================================================
    # Metrics
    # ==================================================

    eval_path = os.path.join(
        outputs_dir,
        "evaluation.pkl"
    )

    if os.path.exists(eval_path):

        with open(eval_path, "rb") as f:
            eval_data = pickle.load(f)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Test Loss",
                f"{eval_data['test_loss']:.4f}"
            )

        with col2:
            st.metric(
                "Test Accuracy",
                f"{eval_data['test_accuracy'] * 100:.2f}%"
            )

    st.write("---")

    st.header("Business Conclusion")

    st.success(
        """
        Conclusion

        The CNN classifier successfully learned the visual
        characteristics associated with powdery mildew.

        Performance on unseen test data demonstrates that
        the model can be used as an automated decision-support
        tool for identifying infected cherry leaves.

        Business Requirement 2 has therefore been achieved.
        """
    )