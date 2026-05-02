from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .utils import load_json


def load_image(image_path: str | Path, image_size: tuple[int, int], rescale: bool) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    if rescale:
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(
    image_path: str | Path,
    model_path: str | Path,
    class_indices_path: str | Path,
    image_size: tuple[int, int] = (224, 224),
    rescale: bool = False,
) -> dict:
    model = tf.keras.models.load_model(model_path)
    class_indices = load_json(class_indices_path)
    classes = {index: label for label, index in class_indices.items()}

    batch = load_image(image_path, image_size=image_size, rescale=rescale)
    probabilities = model.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(probabilities))

    return {
        "label": classes[best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": {classes[i]: float(probabilities[i]) for i in range(len(probabilities))},
    }
