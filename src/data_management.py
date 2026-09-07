import inspect
import os
import pickle
import requests
import joblib

import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

def patch_layer_from_config():
    """
    Patches tf.keras.layers.Layer.from_config to filter out Keras 3
    metadata keys that cause TypeErrors when loaded in Keras 2 (TF 2.15).
    """
    @classmethod
    def patched_from_config(cls, config):
        config_copy = config.copy()
        # Remove metadata keys added by newer Keras versions
        for key in ["build_config", "module", "registered_name", "compile_config"]:
            config_copy.pop(key, None)

        # Inspect layer __init__ signature and filter unrecognized arguments
        try:
            sig = inspect.signature(cls.__init__)
            has_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not has_kwargs:
                valid_params = set(sig.parameters.keys())
                config_copy = {k: v for k, v in config_copy.items() if k in valid_params}
        except Exception:
            pass

        return cls(**config_copy)

    tf.keras.layers.Layer.from_config = patched_from_config

# Apply deserialization patch before loading model
patch_layer_from_config()

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

@st.cache_resource
def load_model_and_classes(model_path, class_indices_path):
    """
    Loads and caches the trained Keras model and class index mapping.
    Automatically downloads the model from GitHub Releases if missing.
    """
    download_model_if_missing(model_path)

    custom_objects = {
        "Rescaling": tf.keras.layers.Rescaling,
        "RandomFlip": tf.keras.layers.RandomFlip,
        "RandomRotation": tf.keras.layers.RandomRotation,
        "RandomZoom": tf.keras.layers.RandomZoom,
        "Conv2D": tf.keras.layers.Conv2D,
        "MaxPooling2D": tf.keras.layers.MaxPooling2D,
        "Flatten": tf.keras.layers.Flatten,
        "Dense": tf.keras.layers.Dense,
        "Dropout": tf.keras.layers.Dropout
    }

    with tf.keras.utils.custom_object_scope(custom_objects):
        model = tf.keras.models.load_model(model_path, compile=False)

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