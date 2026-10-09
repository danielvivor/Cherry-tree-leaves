import os
import pickle
import requests

import streamlit as st
import tensorflow as tf


# ======================================================
# MODEL DOWNLOAD
# ======================================================

def download_model_if_missing(model_path):
    """
    Download trained model from GitHub Releases if missing.
    """

    if (
        os.path.exists(model_path)
        and os.path.getsize(model_path) < 10 * 1024 * 1024
    ):
        os.remove(model_path)

    if not os.path.exists(model_path):

        os.makedirs(
            os.path.dirname(model_path),
            exist_ok=True
        )

        url = (
            "https://github.com/danielvivor/"
            "Cherry-tree-leaves/releases/download/"
            "v1.0.0/"
            "powdery_mildew_detector_model.h5"
        )

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            stream=True,
            allow_redirects=True
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Model download failed with "
                f"HTTP {response.status_code}"
            )

        with open(model_path, "wb") as f:
            for chunk in response.iter_content(
                chunk_size=8192
            ):
                f.write(chunk)


# ======================================================
# MODEL LOADER
# ======================================================

@st.cache_resource
def load_model_and_classes(
    model_path,
    class_indices_path,
):
    """
    Load trained Keras model and class mappings.
    """

    download_model_if_missing(
        model_path
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    with open(
        class_indices_path,
        "rb"
    ) as f:

        class_indices = pickle.load(f)

    map_labels = {
        v: k
        for k, v in class_indices.items()
    }

    return model, map_labels


# ======================================================
# PICKLE LOADER
# ======================================================

@st.cache_data
def load_pkl_data(file_path):
    """
    Load pickle artifacts such as
    training history and evaluation metrics.
    """

    with open(
        file_path,
        "rb"
    ) as f:

        data = pickle.load(f)

    return data


# ======================================================
# DEBUG
# ======================================================

if __name__ == "__main__":
    print(
        "data_management.py loaded successfully"
    )