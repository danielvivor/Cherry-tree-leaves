import os
import pickle
import requests

import streamlit as st
import tensorflow as tf

from tensorflow.keras.layers import (
    InputLayer,
    Rescaling,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)

from tensorflow.keras.mixed_precision import Policy

# =====================================================
# MODEL DOWNLOAD
# =====================================================

def download_model_if_missing(model_path):
    """
    Download the model from GitHub Releases if missing.
    """

    if os.path.exists(model_path) and os.path.getsize(model_path) < 10 * 1024 * 1024:
        os.remove(model_path)

    if not os.path.exists(model_path):

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        url = (
            "https://github.com/danielvivor/"
            "Cherry-tree-leaves/releases/download/"
            "v1.0.0/powdery_mildew_detector_model.h5"
        )

        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, stream=True, allow_redirects=True)

        if response.status_code != 200:
            raise RuntimeError(
                f"Model download failed. " f"HTTP {response.status_code}"
            )

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


# =====================================================
# CONFIG SANITISER
# =====================================================

def sanitize_keras_config(config):
    """
    Remove Keras 3 attributes that Keras 2.15 does not
    understand.
    """

    if not isinstance(config, dict):
        return config

    cleaned = {}

    for key, value in config.items():

        # Remove unsupported keys

        if key in ["optional", "data_format"]:
            continue

        # Convert Keras 3 DTypePolicy

        if key == "dtype" and isinstance(value, dict) and "config" in value:
            cleaned[key] = value["config"].get("name", "float32")

        elif key == "batch_shape" and isinstance(value, list):
            cleaned["input_shape"] = tuple(value[1:])

        elif isinstance(value, dict):
            cleaned[key] = sanitize_keras_config(value)

        elif isinstance(value, list):
            cleaned[key] = [sanitize_keras_config(item) for item in value]

        else:
            cleaned[key] = value

    return cleaned


# =====================================================
# COMPATIBILITY LAYERS
# =====================================================

class FixedInputLayer(InputLayer):

    @classmethod
    def from_config(cls, config):
        config = sanitize_keras_config(config)
        return super().from_config(config)


class FixedRescaling(Rescaling):

    @classmethod
    def from_config(cls, config):
        config = sanitize_keras_config(config)
        return super().from_config(config)


class FixedRandomFlip(RandomFlip):

    @classmethod
    def from_config(cls, config):
        config = sanitize_keras_config(config)
        return super().from_config(config)


class FixedRandomRotation(RandomRotation):

    @classmethod
    def from_config(cls, config):
        config = sanitize_keras_config(config)
        return super().from_config(config)


class FixedRandomZoom(RandomZoom):

    @classmethod
    def from_config(cls, config):
        config = sanitize_keras_config(config)
        return super().from_config(config)


# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_model_and_classes(model_path, class_indices_path):

    download_model_if_missing(model_path)

    custom_objects = {
        "InputLayer": FixedInputLayer,
        "Rescaling": FixedRescaling,
        "RandomFlip": FixedRandomFlip,
        "RandomRotation": FixedRandomRotation,
        "RandomZoom": FixedRandomZoom,
        "DTypePolicy": Policy("float32"),
    }

    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects=custom_objects
    )

    with open(class_indices_path, "rb") as f:

        class_indices = pickle.load(f)

    map_labels = {v: k for k, v in class_indices.items()}

    return model, map_labels


# =====================================================
# PICKLE LOADER
# =====================================================

@st.cache_data
def load_pkl_data(file_path):

    with open(file_path, "rb") as f:

        data = pickle.load(f)

    return data

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":
    print("data_management.py loaded successfully")
