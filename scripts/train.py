from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blood_cell_cancer.config import load_config
from blood_cell_cancer.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a blood cell cancer classifier.")
    parser.add_argument("--data-dir", required=True, help="Path to dataset root with one folder per class.")
    parser.add_argument("--model", default="efficientnetb3", choices=["cnn", "efficientnetb3"], help="Model architecture.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    metrics = train(args.data_dir, args.model, config)
    print(metrics)


if __name__ == "__main__":
    main()
