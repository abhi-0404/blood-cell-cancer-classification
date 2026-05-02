from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blood_cell_cancer.config import load_config
from blood_cell_cancer.data import load_dataset
from blood_cell_cancer.evaluate import evaluate_model, predict_labels, save_classification_report, save_confusion_matrix
from blood_cell_cancer.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier.")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root with one folder per class.")
    parser.add_argument("--model-path", required=True, help="Path to a .keras or .h5 model file.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--rescale", action="store_true", help="Use this for custom CNN models trained on rescaled pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report_dir = ensure_dir(config.report_dir)
    ensure_dir(report_dir / "figures")

    _, _, _, _, generators = load_dataset(args.data_dir, config, rescale=args.rescale)
    _, _, test_gen = generators
    model = tf.keras.models.load_model(args.model_path)

    metrics = evaluate_model(model, test_gen)
    save_json(metrics, report_dir / "evaluation_metrics.json")

    class_names = list(test_gen.class_indices.keys())
    y_true, y_pred = predict_labels(model, test_gen)
    save_classification_report(y_true, y_pred, class_names, report_dir / "evaluation_classification_report.json")
    save_confusion_matrix(y_true, y_pred, class_names, report_dir / "figures" / "evaluation_confusion_matrix.png")
    print(metrics)


if __name__ == "__main__":
    main()
