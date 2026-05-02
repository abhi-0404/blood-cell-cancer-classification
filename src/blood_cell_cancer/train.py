from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf

from .config import TrainingConfig
from .data import load_dataset
from .evaluate import evaluate_model, predict_labels, save_classification_report, save_confusion_matrix
from .models import build_model, unfreeze_efficientnet
from .utils import ensure_dir, save_json, set_seed


def make_callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
    ]


def train(data_dir: str | Path, model_name: str, config: TrainingConfig) -> dict:
    set_seed(config.random_state)
    model_dir = ensure_dir(config.model_dir)
    report_dir = ensure_dir(config.report_dir)
    ensure_dir(report_dir / "figures")

    rescale = model_name.lower() == "cnn"
    df, train_df, valid_df, test_df, generators = load_dataset(data_dir, config, rescale=rescale)
    train_gen, valid_gen, test_gen = generators
    class_names = list(train_gen.class_indices.keys())

    save_json(train_gen.class_indices, model_dir / "class_indices.json")
    pd.concat(
        [
            train_df.assign(split="train"),
            valid_df.assign(split="validation"),
            test_df.assign(split="test"),
        ],
        ignore_index=True,
    ).to_csv(report_dir / "data_split.csv", index=False)

    model_path = model_dir / f"best_{model_name.lower()}.keras"
    model = build_model(model_name, config.input_shape, len(class_names), config.learning_rate)
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=config.epochs,
        callbacks=make_callbacks(model_path),
    )

    if model_name.lower() in {"efficientnet", "efficientnetb3", "effb3"} and config.fine_tune_epochs > 0:
        model = unfreeze_efficientnet(model, learning_rate=config.fine_tune_learning_rate)
        model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=config.fine_tune_epochs,
            callbacks=make_callbacks(model_path),
        )

    model.save(model_path)

    metrics = {
        "dataset_size": int(len(df)),
        "train": evaluate_model(model, train_gen),
        "validation": evaluate_model(model, valid_gen),
        "test": evaluate_model(model, test_gen),
    }
    save_json(metrics, report_dir / f"{model_name.lower()}_metrics.json")

    y_true, y_pred = predict_labels(model, test_gen)
    save_classification_report(y_true, y_pred, class_names, report_dir / f"{model_name.lower()}_classification_report.json")
    save_confusion_matrix(y_true, y_pred, class_names, report_dir / "figures" / f"{model_name.lower()}_confusion_matrix.png")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(report_dir / f"{model_name.lower()}_history.csv", index=False)
    return metrics
