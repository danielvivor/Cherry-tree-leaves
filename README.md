# 🍃 Mildew Detection in Cherry Leaves

The **Mildew Detection in Cherry Leaves** application is an end-to-end Machine Learning tool built with Python and Streamlit to visually detect powdery mildew in cherry tree leaf samples. It aims to streamline crop inspection for **Farmy & Foods** by replacing a time-consuming manual verification process with an instant, scalable ML prediction engine.

---

## 🎯 Business Requirements
- **Business Requirement 1:** Conduct a visual study to differentiate healthy cherry leaves from leaves infected with powdery mildew using average images, variability plots, difference images, and visual montages.
- **Business Requirement 2:** Deliver an accurate binary classification model predicting with at least **97% accuracy** whether a cherry leaf image is healthy or contains powdery mildew.

## 👤 User Stories & ML Mapping

| User Story | Description | Task Type | Action & Deliverable | Mapping |
| :--- | :--- | :--- | :--- | :--- |
| **1** | As a client, I want to view average image characteristics, variability, and visual differences between healthy and infected leaves, so that I can visually distinguish between them. | Data Visualisation | Interactive **Leaf Visualizer** page displaying average images, variability plots, difference images, and visual montages. | Requirement 1 |
| **2** | As a client/farm operator, I want to upload single or batch leaf images, so that I can predict instantly whether a leaf is healthy or infected with powdery mildew. | Machine Learning | Interactive **Mildew Detector** page featuring an image uploader widget, real-time CNN prediction engine, confidence scores, and downloadable CSV reports. | Requirement 2 |
| **3** | As an analyst/IT lead, I want to view ML model performance metrics (loss/accuracy curves, confusion matrix), so that I can verify model accuracy and trust its predictions before scaling. | ML Evaluation | Dedicated **ML Performance** page showcasing dataset distribution, training history plots, held-out test evaluation, and formal validation statements. | Requirements 1 & 2 |

## 🔬 Machine Learning Business Case

* **Goal:** Predict whether a given cherry leaf image is healthy or infected with powdery mildew.
* **Learning Method:** Supervised Binary Classification using a Convolutional Neural Network (CNN) built in TensorFlow/Keras.
* **Ideal Outcome:** An automated, scalable Streamlit dashboard that replaces 30-minute manual tree inspections with instant digital predictions (< 2 seconds), allowing staff to target precise chemical treatments.
* **Data & Splits:** 4,208 cherry leaf images split into **Train (70%)**, **Validation (10%)**, and **Test (20%)** sets, with data augmentation applied during training.
* **Output:** Categorical label (`Healthy` or `Powdery Mildew`) alongside a continuous prediction probability score and percentage confidence.
* **Success Criteria:** Achieve at least 97% accuracy and high recall on the held-out test dataset.

## 📈 Model Performance & Evaluation Results

The trained CNN model surpassed the business criteria target (>97% accuracy):

* **Test Accuracy:** **100.00%** across 844 test set images.
* **Test Loss:** `0.0000` (negligible categorical cross-entropy loss).
* **Confusion Matrix:** Perfect classification on the test set—**422/422 healthy** and **422/422 powdery mildew** leaves were correctly identified with zero false positives or false negatives.
* **Training Dynamics:** Both training and validation accuracy stabilized above 99% past epoch 8, while validation loss dropped smoothly to near zero without overfitting.

## 🖥️ Dashboard Architecture & Design

The dashboard is organized into five interactive pages:

1. **Page 1: Summary** — High-level project background, dataset details, CRISP-DM methodology, and business requirements mapping.
2. **Page 2: Leaf Visualizer** — Interactive visual study displaying average images, standard deviation plots, and image differences.
3. **Page 3: Mildew Detector** — File uploader widget supporting single or batch uploads, real-time CNN prediction results, confidence meters, and downloadable prediction summary CSV reports.
4. **Page 4: Hypotheses & Validation** — Explanations of visual signatures (e.g., white fungal patches) verified through pixel intensity analysis.
5. **Page 5: ML Performance** — Training history plots (Accuracy & Loss curves), test set confusion matrix, and performance metrics.

## 📁 Project Directory Structure

```text
├── app.py                      # Main Streamlit dashboard application
├── src/                        # Modular source code
│   ├── data_management.py      # Artifact loading & caching utilities
│   └── machine_learning.py     # Image preprocessing & inference engine
├── outputs/v1/                 # Saved model artifacts & visualization outputs
│   ├── powdery_mildew_detector_model.h5
│   ├── class_indices.pkl
│   ├── evaluation.pkl
│   ├── avg_var_healthy.png
│   ├── avg_var_powdery_mildew.png
│   ├── avg_diff.png
│   ├── confusion_matrix.png
│   └── model_training_history.png
├── inputs/                     # Kaggle raw dataset & split image directories
├── Procfile                    # Deployment execution command
├── setup.sh                    # Streamlit server port configuration script
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Specified Python version (3.10.12)
└── README.md                   # Project documentation
````

## 🛠️ Technologies Used
- Language: Python 3.10.12

- Web Framework: Streamlit

- Deep Learning: TensorFlow, Keras

- Data Processing & Manipulation: NumPy, Pandas

- Data Visualization: Matplotlib, Seaborn

- Image Processing: Pillow (PIL)

- Model Serialization: Pickle, Joblib

## 🚀 Local Setup & Deployment
Local Execution  

1. Clone the repository:  
`Bash
git clone <repository-url>
cd <repository-folder>`

2. Create and activate a virtual environment:  
`Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate`

3. Install dependencies:  
`Bash
pip install -r requirements.txt`


5. Launch the Streamlit dashboard:  
`Bash
streamlit run app.py`

## Cloud Deployment (Heroku)
The repository includes all required deployment configuration files:

- `Procfile`: Executes `setup.sh` and runs `app.py`.

- `setup.sh`: Dynamically binds Streamlit to the `$PORT` provided by the cloud platform.

- `requirements.txt`: Specifies `tensorflow-cpu` to keep slug sizes light and optimize container startup.

- `runtime.txt`: Pins the Python runtime engine (`python-3.10.12`).


