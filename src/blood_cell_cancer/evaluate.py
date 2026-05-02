from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .utils import ensure_dir, save_json


def evaluate_model(model: tf.keras.Model, generator) -> dict[str, float]:
    loss, accuracy = model.evaluate(generator, verbose=0)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def predict_labels(model: tf.keras.Model, generator) -> tuple[np.ndarray, np.ndarray]:
    generator.reset()
    probabilities = model.predict(generator, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = generator.classes
    return y_true, y_pred


def save_classification_report(y_true, y_pred, class_names: list[str], output_path: str | Path) -> dict:
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    save_json(report, output_path)
    return report


def save_confusion_matrix(y_true, y_pred, class_names: list[str], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
