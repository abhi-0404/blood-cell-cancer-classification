from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    image_size: tuple[int, int] = (224, 224)
    channels: int = 3
    batch_size: int = 32
    epochs: int = 20
    fine_tune_epochs: int = 10
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    train_split: float = 0.70
    validation_split: float = 0.15
    test_split: float = 0.15
    random_state: int = 42
    exclude_classes: list[str] = field(default_factory=lambda: ["ig", "neutrophil"])
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (*self.image_size, self.channels)


def _coerce_value(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        return [_coerce_value(item) for item in value[1:-1].split(",") if item.strip()]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _load_yaml_like(path: Path) -> dict[str, Any]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except ModuleNotFoundError:
        raw: dict[str, Any] = {}
        current_list_key: str | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("-") and current_list_key:
                raw[current_list_key].append(_coerce_value(stripped[1:]))
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                raw[key] = _coerce_value(value)
                current_list_key = None
            else:
                raw[key] = []
                current_list_key = key
        return raw


def load_config(path: str | Path = "configs/default.yaml") -> TrainingConfig:
    config_path = Path(path)
    if not config_path.exists():
        return TrainingConfig()

    raw = _load_yaml_like(config_path)

    if "image_size" in raw:
        raw["image_size"] = tuple(raw["image_size"])
    if "model_dir" in raw:
        raw["model_dir"] = Path(raw["model_dir"])
    if "report_dir" in raw:
        raw["report_dir"] = Path(raw["report_dir"])

    allowed = TrainingConfig.__dataclass_fields__.keys()
    filtered: dict[str, Any] = {key: value for key, value in raw.items() if key in allowed}
    return TrainingConfig(**filtered)
