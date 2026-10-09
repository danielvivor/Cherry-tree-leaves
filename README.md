# 🍃 Mildew Detection in Cherry Leaves

The **Mildew Detection in Cherry Leaves** application is an end-to-end Machine Learning solution developed using **Python**, **TensorFlow/Keras**, and **Streamlit**. The system enables rapid identification of powdery mildew disease in cherry tree leaves through image classification, helping Farmy & Foods replace a slow, labour-intensive manual inspection process with an automated and scalable solution.

---

# Business Understanding (CRISP-DM)

## Business Requirements

### Business Requirement 1

Conduct a visual study to differentiate healthy cherry leaves from leaves infected with powdery mildew.

### Business Requirement 2

Develop a Machine Learning solution capable of predicting whether a cherry leaf is healthy or infected with powdery mildew with a minimum accuracy threshold of **97%**.

---

# Dataset Content

## Dataset Source

The dataset used in this project was sourced from the Code Institute Cherry Leaves dataset available on Kaggle:

https://www.kaggle.com/codeinstitute/cherry-leaves

## Dataset Description

The dataset contains images of cherry leaves belonging to two classes:

- Healthy
- Powdery Mildew

The images were collected to support the development of an automated disease detection system capable of identifying fungal infection in cherry plantations.

## Dataset Characteristics

| Feature | Value |
|----------|----------|
| Data Type | Image Dataset |
| Classification Type | Binary Classification |
| Number of Classes | 2 |
| Image Colour Space | RGB |
| Input Dimensions | 256 × 256 × 3 |
| Target Variable | Leaf Health Status |

## Target Classes

### Healthy

- 2,104 images
- Leaves free from visible disease symptoms

### Powdery Mildew

- 2,104 images
- Leaves displaying fungal infection symptoms including white powder-like growth

## Dataset Split

| Dataset | Images |
|----------|----------|
| Training | 2,944 |
| Validation | 422 |
| Test | 842 |
| Total | 4,208 |

This train-validation-test split was implemented to ensure robust model evaluation and prevent data leakage.

---

# Rationale to Map Business Requirements to Data Visualisations and ML Tasks

## Business Requirement 1

### Objective

Identify visual differences between healthy leaves and leaves infected with powdery mildew.

### Data Visualisation Tasks

The following visual studies were performed:

- Healthy leaf image montage
- Powdery mildew image montage
- Average image analysis
- Variability image analysis
- Difference between average images
- Pixel-intensity distribution analysis
- Image dimension analysis
- Dataset class distribution analysis

### Outcome

The visualisations revealed distinguishable patterns between healthy leaves and leaves affected by powdery mildew, supporting the feasibility of automated image classification.

---

## Business Requirement 2

### Objective

Predict whether a cherry leaf contains powdery mildew.

### Machine Learning Task

**Binary Image Classification**

### Learning Method

Supervised Deep Learning using a Convolutional Neural Network (CNN).

### Input

Cherry leaf image.

### Output

Prediction label:

- Healthy
- Powdery Mildew

along with prediction probability and confidence score.

### Outcome

An automated prediction system integrated into the Streamlit dashboard.

---

# User Stories

| User Story | Requirement |
|------------|-------------|
| As a client, I want to visually compare healthy and infected leaves so I can understand the characteristics of powdery mildew. | Business Requirement 1 |
| As a farm operator, I want to upload an image and receive a disease prediction instantly. | Business Requirement 2 |
| As a decision-maker, I want to evaluate model performance before deploying the solution. | Business Requirement 2 |

---

# Machine Learning Business Case

## Goal

Predict whether a cherry leaf is healthy or infected with powdery mildew.

## Learning Method

Supervised Binary Classification using a Convolutional Neural Network (CNN).

## Ideal Outcome

Replace manual inspection processes with a near-instant automated diagnostic tool that significantly reduces inspection time while maintaining high diagnostic accuracy.

## Model Inputs

- Cherry leaf image (256 × 256 × 3)

## Model Outputs

- Healthy
- Powdery Mildew

Confidence probability is also returned to assist decision-making.

## Success Criteria

- Achieve a minimum of 97% test accuracy.
- Demonstrate strong generalisation on unseen data.
- Maintain low test loss.

## Failure Criteria

- Test accuracy below 97%.
- Significant divergence between training and validation performance.
- Excessive false positives or false negatives.

---

# Project Hypotheses

## Hypothesis 1

Cherry leaves affected by powdery mildew contain visual characteristics that can be objectively distinguished from healthy leaves through image analysis.

### Validation

Evidence collected from:

- Average image analysis
- Variability image analysis
- Difference-image analysis
- Leaf montages
- Pixel-intensity distribution analysis

### Conclusion

The visual study consistently revealed distinguishable differences between healthy and infected leaves.

**Hypothesis Supported.**

---

## Hypothesis 2

A Convolutional Neural Network can classify cherry leaf images with an accuracy greater than 97%.

### Validation

The trained CNN model was evaluated on previously unseen test images.

Evaluation included:

- Test Accuracy
- Test Loss
- Confusion Matrix
- Learning Curve Analysis

### Conclusion

The model exceeded the project success criterion of 97% accuracy.

**Hypothesis Supported.**

---

# Data Analysis Findings

## Finding 1

Average image analysis revealed visible differences between healthy leaves and leaves affected by powdery mildew.

## Finding 2

Difference-image analysis highlighted regions where infected leaves differed consistently from healthy leaves.

## Finding 3

Image montages demonstrated recurring fungal patterns within infected samples.

## Finding 4

Pixel-intensity distributions showed measurable colour-space differences between classes.

### Overall Conclusion

Business Requirement 1 was successfully addressed through visual analysis.

---

# Model Training & Evolution

Several model configurations were explored before selecting the final architecture.

## Experiment 1

- Baseline CNN architecture
- Initial performance benchmark established

## Experiment 2

- Image augmentation introduced
- Improved model generalisation

## Experiment 3

- Dropout layers implemented
- Early Stopping introduced
- Reduced risk of overfitting

## Final Model

- CNN architecture
- Data augmentation enabled
- Dropout regularisation enabled
- Early Stopping enabled
- Selected based on strongest validation performance

---

# Model Performance

## Evaluation Results

| Metric | Result |
|----------|----------|
| Test Accuracy | 100.00% |
| Test Loss | 0.0000 |

## Confusion Matrix Results

The confusion matrix demonstrated perfect classification performance on the held-out test dataset.

- Healthy leaves correctly classified
- Powdery mildew leaves correctly classified
- No observed misclassifications within the test data

While these results indicate exceptional performance, future evaluation using newly collected field images would provide additional confirmation of real-world generalisation.

## Learning Curves

Training and validation accuracy improved together while training and validation loss decreased consistently.

The close alignment between the curves indicates strong generalisation and minimal evidence of overfitting.

## Final Verdict

The model exceeded the required success threshold of **97% accuracy** and therefore successfully satisfies **Business Requirement 2**.

---

# Dashboard Design

## Summary Page

### Content

- Project overview
- Dataset information
- Dataset metrics
- Business requirements

### Addresses

- Business Understanding
- CRISP-DM Business Understanding

---

## Leaf Visualizer Page

### Content

- Healthy montage
- Powdery mildew montage
- Average images
- Variability images
- Difference image
- Pixel-intensity histogram
- Image dimension plot
- Class-distribution plot
- Plot interpretations

### Addresses

- Business Requirement 1

---

## Powdery Mildew Detector Page

### Content

- Image uploader
- Live predictions
- Confidence scores
- Prediction summary table
- CSV report download

### Addresses

- Business Requirement 2

---

## Project Hypotheses Page

### Content

- Hypotheses
- Validation procedures
- Supporting evidence
- Conclusions

### Addresses

- Business Requirements 1 and 2

---

## ML Performance Page

### Content

- Model evolution
- Hyperparameter discussion
- Interactive learning curves
- Confusion matrix
- Evaluation metrics
- Final model verdict

### Addresses

- Business Requirement 2

---

# Testing

## Manual Application Testing

| Test Case | Expected Result | Outcome |
|------------|----------------|----------|
| Summary Page Loads | Dataset information displayed | Pass |
| Leaf Visualizer Loads | Visualisations displayed successfully | Pass |
| Hypotheses Page Loads | Validation information displayed | Pass |
| ML Performance Loads | Evaluation information displayed | Pass |
| Detector Accepts Images | Uploaded images processed | Pass |
| CSV Export Works | Prediction report downloads | Pass |

## Model Testing

The final CNN model was evaluated using:

- Test Accuracy
- Test Loss
- Confusion Matrix
- Training History
- Validation History

The model exceeded the business success criterion.

---

# Project Structure

```text
├── app.py
├── app_pages
│   ├── page_summary.py
│   ├── page_leaf_visualizer.py
│   ├── page_mildew_detector.py
│   ├── page_project_hypotheses.py
│   └── page_ml_performance.py
│
├── src
│   ├── data_management.py
│   └── machine_learning.py
│
├── outputs
├── inputs
├── requirements.txt
├── runtime.txt
├── Procfile
├── setup.sh
└── README.md
```

---

# Technologies Used

## Languages

- Python

## Frameworks

- Streamlit
- TensorFlow
- Keras

## Data Processing

- NumPy
- Pandas

## Visualisation

- Plotly
- Matplotlib
- Seaborn

## Image Processing

- Pillow

## Deployment

- Heroku

---

# Deployment

## Local Deployment

```bash
git clone <repository-url>

cd <repository-folder>

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

### Windows

```bash
.venv\Scripts\activate

streamlit run app.py
```

## Heroku Deployment

The project includes:

- `Procfile`
- `runtime.txt`
- `requirements.txt`
- `setup.sh`

These files allow deployment of the Streamlit dashboard to Heroku.

---

# Credits

## Dataset

- Code Institute Cherry Leaves Dataset
- Kaggle

## Libraries

- TensorFlow
- Keras
- Streamlit
- Plotly
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow

## Acknowledgements

- Code Institute
- Farmy & Foods project brief
- Cherry Leaves Machine Learning walkthrough inspiration









