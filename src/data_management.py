import os
import pickle
import requests
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Rescaling
from tensorflow.keras.mixed_precision import Policy


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


def sanitize_keras_config(config):
    """
    Recursively strips Keras 3 specific parameters ('DTypePolicy', 'optional', 'batch_shape')
    from layer configurations so Keras 2 / TF 2.15 can deserialize the model cleanly.
    """
    if not isinstance(config, dict):
        return config

    cleaned = {}
    for key, value in config.items():
        # Handle Keras 3 DTypePolicy dictionary structure
        if key == "dtype" and isinstance(value, dict) and "config" in value:
            cleaned[key] = value["config"].get("name", "float32")
        # Remove unsupported Keras 3 attributes on InputLayer
        elif key == "optional":
            continue
        elif key == "batch_shape" and isinstance(value, list) and len(value) > 1:
            cleaned["input_shape"] = tuple(value[1:])
        elif isinstance(value, dict):
            cleaned[key] = sanitize_keras_config(value)
        elif isinstance(value, list):
            cleaned[key] = [sanitize_keras_config(item) for item in value]
        else:
            cleaned[key] = value

    return cleaned


class FixedInputLayer(InputLayer):
    @classmethod
    def from_config(cls, config):
        return super().from_config(sanitize_keras_config(config))


class FixedRescaling(Rescaling):
    @classmethod
    def from_config(cls, config):
        return super().from_config(sanitize_keras_config(config))


@st.cache_resource
def load_model_and_classes(model_path, class_indices_path):
    """
    Loads and caches the trained Keras model, mapping Keras 3 config types to compatible layers.
    """
    download_model_if_missing(model_path)

    custom_objects = {
        "InputLayer": FixedInputLayer,
        "Rescaling": FixedRescaling,
        "DTypePolicy": Policy("float32"),
    }

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=custom_objects
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