# Page 4: Project Hypotheses & Validation
import streamlit as st

def page_project_hypotheses():

    st.title("🔬 Project Hypotheses & Validation")

    st.write("""
    This page evaluates the project hypotheses and presents the evidence
    collected during data analysis and model evaluation.
    """)

    st.write("---")

    # =====================================================
    # Hypothesis 1
    # =====================================================

    st.header("Hypothesis 1")

    st.markdown("""
    **Hypothesis**

    Cherry leaves affected by powdery mildew exhibit visual patterns
    that can be objectively differentiated from healthy leaves using
    image analysis techniques.
    """)

    st.subheader("Evidence")

    st.markdown("""
    - Average image analysis revealed visible differences between healthy and infected leaves.
    - Variability images showed recurring visual patterns across infected samples.
    - Difference-image analysis highlighted regions where the two classes diverged consistently.
    - Leaf montages demonstrated that powdery mildew leaves repeatedly contain light-coloured fungal markings that are not present in healthy leaves.
    """)

    st.success("""
    Validation Result

    The visual study provided objective evidence that healthy and
    powdery mildew leaves contain distinguishable visual features.

    Therefore, Hypothesis 1 is supported.
    """)

    st.write("---")

    # =====================================================
    # Hypothesis 2
    # =====================================================

    st.header("Hypothesis 2")

    st.markdown("""
    **Hypothesis**

    A Convolutional Neural Network (CNN) can accurately classify
    cherry leaf images as Healthy or Powdery Mildew and exceed
    the business success criterion of 97% accuracy.
    """)

    st.subheader("Evidence")

    st.markdown("""
    - Model training demonstrated stable convergence.
    - Training and validation learning curves remained closely aligned.
    - The confusion matrix showed extremely strong classification performance.
    - Evaluation on unseen test data exceeded the business performance target.
    """)

    st.success("""
    Validation Result

    The final model exceeded the required accuracy threshold of 97%.

    Therefore, Hypothesis 2 is supported and Business Requirement 2
    has been successfully satisfied.
    """)

    st.write("---")

    st.header("Overall Findings")

    st.info("""
    Conclusion

    The visual analysis confirmed that powdery mildew infection produces
    identifiable image features.

    The machine learning pipeline successfully learned these patterns
    and demonstrated performance that satisfies the project's business
    requirements.
    """)