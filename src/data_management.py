import os
import pickle
import requests
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import InputLayer


def download_model_if_missing(model_path):
    """
    Downloads the trained .h5 model from GitHub Releases if missing or invalid.
    """
    if os.path.exists(model_path) and os.path.getsize(model_path) < 10 * 1024 * 1024:
        os.remove(model_path)

    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        url = (
            "https://github.com/danielvivor/Cherry-tree-leaves/releases/download/"
            "v1.0.0/powdery_mildew_detector_model.h5"
        )

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, stream=True, allow_redirects=True)

        if response.status_code != 200:
            raise RuntimeError(
                f"Model download failed with HTTP Status {response.status_code} for URL: {url}. "
                "Ensure release 'v1.0.0' is Published and repository is Public."
            )

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


# Custom InputLayer wrapper to safely sanitize legacy/Keras 3 config keys
class FixedInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        config = config.copy()
        # Clean unrecognized attributes
        config.pop("optional", None)
        if "batch_shape" in config:
            batch_shape = config.pop("batch_shape")
            if batch_shape and len(batch_shape) > 1:
                config["input_shape"] = tuple(batch_shape[1:])
        return super().from_config(config)


@st.cache_resource
def load_model_and_classes(model_path, class_indices_path):
    """
    Loads and caches the trained Keras model, passing custom objects to safely deserialize InputLayer config.
    """
    download_model_if_missing(model_path)

    # Load model passing custom object mapping to handle modified InputLayer schema
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"InputLayer": FixedInputLayer}
    )

    with open(class_indices_path, "rb") as f:
        class_indices = pickle.load(f)

    map_labels = {v: k for k, v in class_indices.items()}
    return model, map_labels


@st.cache_data
def load_pkl_data(file_path):
    """
    Loads pickle artifacts such as evaluation metrics or training history.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    print("data_management.py executed successfully!")