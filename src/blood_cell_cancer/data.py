from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import TrainingConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def validate_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {path}")

    class_dirs = [item for item in path.iterdir() if item.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found in dataset directory: {path}")
    return path


def build_dataframe(data_dir: str | Path, exclude_classes: list[str] | None = None) -> pd.DataFrame:
    root = validate_data_dir(data_dir)
    excluded = {label.lower() for label in (exclude_classes or [])}
    rows: list[dict[str, str]] = []

    for class_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        label = class_dir.name
        if label.lower() in excluded:
            continue

        for image_path in class_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"filepaths": str(image_path), "labels": label})

    if not rows:
        raise ValueError(f"No image files found in {root}. Supported extensions: {sorted(IMAGE_EXTENSIONS)}")

    return pd.DataFrame(rows)


def split_dataframe(df: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if round(config.train_split + config.validation_split + config.test_split, 5) != 1:
        raise ValueError("train_split + validation_split + test_split must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=config.train_split,
        random_state=config.random_state,
        stratify=df["labels"],
    )

    validation_ratio = config.validation_split / (config.validation_split + config.test_split)
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=validation_ratio,
        random_state=config.random_state,
        stratify=temp_df["labels"],
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_generators(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainingConfig,
    rescale: bool,
):
    rescale_value = 1.0 / 255 if rescale else None
    train_datagen = ImageDataGenerator(
        rescale=rescale_value,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(rescale=rescale_value)

    common = {
        "x_col": "filepaths",
        "y_col": "labels",
        "target_size": config.image_size,
        "class_mode": "categorical",
        "color_mode": "rgb",
        "batch_size": config.batch_size,
    }

    train_gen = train_datagen.flow_from_dataframe(train_df, shuffle=True, seed=config.random_state, **common)
    valid_gen = eval_datagen.flow_from_dataframe(valid_df, shuffle=False, **common)
    test_gen = eval_datagen.flow_from_dataframe(test_df, shuffle=False, **common)
    return train_gen, valid_gen, test_gen


def load_dataset(data_dir: str | Path, config: TrainingConfig, rescale: bool):
    df = build_dataframe(data_dir, config.exclude_classes)
    train_df, valid_df, test_df = split_dataframe(df, config)
    generators = make_generators(train_df, valid_df, test_df, config, rescale=rescale)
    return df, train_df, valid_df, test_df, generators
