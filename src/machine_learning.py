# Preprocessing & model inference functions

import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.utils import img_to_array

def load_and_preprocess_image(img_pil, target_size=(256, 256)):
    """
    Resizes PIL image and converts it into a 4D tensor [1, 256, 256, 3].
    Note: Rescaling by 1/255 is handled internally by the model's Rescaling layer.
    """
    img_resized = img_pil.resize(target_size)
    img_array = img_to_array(img_resized)
    img_tensor = np.expand_dims(img_array, axis=0)
    return img_tensor

def predict_leaf(img_pil, model, map_labels, target_size=(256, 256)):
    """
    Runs model inference on a PIL Image object and returns predictions.
    """
    img_tensor = load_and_preprocess_image(img_pil, target_size=target_size)
    pred_prob = model.predict(img_tensor, verbose=0)[0][0]

    if pred_prob > 0.5:
        pred_class = map_labels[1]  # 'powdery_mildew'
        confidence = pred_prob * 100
    else:
        pred_class = map_labels[0]  # 'healthy'
        confidence = (1 - pred_prob) * 100

    return {
        'Diagnostic': pred_class.replace('_', ' ').title(),
        'Confidence (%)': round(confidence, 2),
        'Raw Probability': round(float(pred_prob), 4)
    }