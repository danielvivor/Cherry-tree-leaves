
import os
import pickle
import urllib.request

import pandas as pd
import numpy as np
from PIL import Image

import streamlit as st

# Auto-download model if missing
def download_model_if_missing(model_path):
    """
    Downloads the trained .h5 model from GitHub Releases if missing or invalid.
    Removes corrupted or tiny files.
    """
    # Remove invalid or incomplete files (<10 MB)
    if os.path.exists(model_path) and os.path.getsize(model_path) < 10 * 1024 * 1024:
        os.remove(model_path)

    # Download if missing
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Direct GitHub Releases asset URL (reliable for Heroku)
        url = (
            "https://github.com/danielvivor/Cherry-tree-leaves/releases/download/"
            "v1.0.0/powdery_mildew_detector_model.h5"
        )

        urllib.request.urlretrieve(url, model_path)

# Load Model + Class Indices
@st.cache_resource
def load_model_and_classes(model_path, class_indices_path):
    """
    Loads and caches the trained Keras model and class index mapping.
    Automatically downloads the model from GitHub if missing.
    """
    import tensorflow as tf
    from keras.models import load_model

    # Ensure model exists locally
    download_model_if_missing(model_path)

    # Load model
    model = load_model(model_path)

    # Load class index mapping
    with open(class_indices_path, 'rb') as f:
        class_indices = pickle.load(f)

    # Invert mapping: {0: 'healthy', 1: 'powdery_mildew'}
    map_labels = {v: k for k, v in class_indices.items()}

    return model, map_labels

# Load Generic Pickle Artifacts
@st.cache_data
def load_pkl_data(file_path):
    """
    Loads pickle artifacts such as evaluation metrics or training history.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Module Execution Check
if __name__ == "__main__":
    print("data_management.py executed successfully! All functions imported without errors.")
