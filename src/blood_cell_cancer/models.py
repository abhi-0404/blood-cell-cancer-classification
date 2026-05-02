from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Sequential


def build_cnn(input_shape: tuple[int, int, int], class_count: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    model = Sequential(
        [
            tf.keras.Input(shape=input_shape),
            Conv2D(32, 3, padding="same", activation="relu"),
            BatchNormalization(),
            Conv2D(32, 3, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.25),
            Conv2D(64, 3, padding="same", activation="relu"),
            BatchNormalization(),
            Conv2D(64, 3, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.30),
            Conv2D(128, 3, padding="same", activation="relu"),
            BatchNormalization(),
            Conv2D(128, 3, padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.35),
            GlobalAveragePooling2D(),
            Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_efficientnetb3(
    input_shape: tuple[int, int, int],
    class_count: int,
    learning_rate: float = 1e-3,
    train_base: bool = False,
) -> tf.keras.Model:
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    base_model.trainable = train_base

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.45)(x)
    outputs = Dense(class_count, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="efficientnetb3_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_efficientnet(model: tf.keras.Model, learning_rate: float = 1e-5, unfreeze_from: int = -40) -> tf.keras.Model:
    base_model = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model)), None)
    if base_model is None:
        raise ValueError("Could not find nested EfficientNet base model.")

    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(model_name: str, input_shape: tuple[int, int, int], class_count: int, learning_rate: float) -> tf.keras.Model:
    normalized = model_name.lower()
    if normalized == "cnn":
        return build_cnn(input_shape, class_count, learning_rate)
    if normalized in {"efficientnet", "efficientnetb3", "effb3"}:
        return build_efficientnetb3(input_shape, class_count, learning_rate)
    raise ValueError(f"Unsupported model: {model_name}. Choose 'cnn' or 'efficientnetb3'.")
