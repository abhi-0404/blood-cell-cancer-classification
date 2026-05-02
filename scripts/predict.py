from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blood_cell_cancer.config import load_config
from blood_cell_cancer.predict import predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one blood cell image.")
    parser.add_argument("--image", required=True, help="Image path.")
    parser.add_argument("--model-path", required=True, help="Path to a .keras or .h5 model file.")
    parser.add_argument("--class-indices", default="models/class_indices.json", help="Path to class index JSON.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--rescale", action="store_true", help="Use this for custom CNN models trained on rescaled pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = predict_image(args.image, args.model_path, args.class_indices, config.image_size, rescale=args.rescale)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
