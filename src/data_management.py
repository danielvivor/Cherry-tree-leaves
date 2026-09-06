# Helper functions to load images & metrics

import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

@st.cache_resource
def load_model_and_classes(model_path, class_indices_path):
    """
    Loads and caches the trained Keras model and class indices dictionary.
    """
    import tensorflow as tf
    from keras.models import load_model

    model = load_model(model_path)
    
    with open(class_indices_path, 'rb') as f:
        class_indices = pickle.load(f)
        
    map_labels = {v: k for k, v in class_indices.items()}
    return model, map_labels

@st.cache_data
def load_pkl_data(file_path):
    """
    Loads generic pickle artifacts (evaluation metrics, training history).
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    print("data_management.py executed successfully! All functions imported without errors.")