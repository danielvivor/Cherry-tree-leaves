import os
import pickle
import requests
import joblib

import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model

import streamlit as st


def download_model_if_missing(model_path):
    """
    Downloads the trained .h5 model from GitHub Releases if missing or invalid.
    Removes corrupted or tiny files (<10MB).
    """
    # Delete invalid/corrupted files (<10 MB)
    if os.path.exists(model_path) and os.path.getsize(model_path) < 10 * 1024 * 1024:
        os.remove(model_path)

    # Download missing model
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Ensure the tag ('v1.0' or 'v1.0.0') matches your GitHub Release tag exactly
        url = (
            "https://github.com/danielvivor/Cherry-tree-leaves/releases/download/"
            "v1.0/powdery_mildew_detector_model.h5"
        )

        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


@st.cache_resource
def load_model_and_classes(model_path, class_indices_path):
    """
    Loads and caches the trained Keras model and class index mapping.
    Downloads the model from GitHub Releases if missing.
    """
    # Ensure model exists locally or gets downloaded
    download_model_if_missing(model_path)

    # Load model
    model = load_model(model_path)

    # Load class index mapping
    with open(class_indices_path, 'rb') as f:
        class_indices = pickle.load(f)

    # Invert mapping: {0: 'healthy', 1: 'powdery_mildew'}
    map_labels = {v: k for k, v in class_indices.items()}

    return model, map_labels


@st.cache_data
def load_pkl_data(file_path):
    """
    Loads pickle artifacts such as evaluation metrics or training history.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    print("data_management.py executed successfully!")