# 🩸 Blood Cell Cancer Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Deep learning pipeline for blood cell cancer image classification using a custom CNN and EfficientNetB3 transfer learning, with a production-ready Streamlit interface.**

</div>

---

## 📌 Overview

This project classifies microscopy blood cell images into 6 clinically relevant categories using two deep learning architectures:

| Model | Type | Strategy |
|---|---|---|
| **Custom CNN** | Baseline | Built from scratch with Conv2D, BatchNorm, Dropout |
| **EfficientNetB3** | Transfer Learning | ImageNet pretrained + fine-tuning |

The full ML workflow is covered end-to-end:

```
Dataset → Preprocessing → Augmentation → Training → Fine-tuning → Evaluation → Prediction → UI
```

---

## 🗂️ Project Structure

```text
blood-cell-cancer-classification/
├── app.py                          # Streamlit web interface
├── configs/
│   └── default.yaml                # Hyperparameters and training config
├── data/
│   └── .gitkeep                    # Place dataset here (data/raw/)
├── models/
│   └── .gitkeep                    # Trained model artifacts saved here
├── notebooks/
│   └── blood-cell-cancer-using-cnn-and-efficientnetb3.ipynb
├── reports/
│   ├── .gitkeep
│   └── figures/
│       └── .gitkeep                # Confusion matrix plots saved here
├── scripts/
│   ├── train.py                    # CLI: train a model
│   ├── evaluate.py                 # CLI: evaluate a trained model
│   └── predict.py                  # CLI: predict a single image
├── src/
│   └── blood_cell_cancer/
│       ├── __init__.py
│       ├── config.py               # TrainingConfig dataclass + YAML loader
│       ├── data.py                 # Dataset loading, splitting, generators
│       ├── models.py               # CNN and EfficientNetB3 builders
│       ├── train.py                # Full training loop with fine-tuning
│       ├── evaluate.py             # Metrics, confusion matrix, reports
│       ├── predict.py              # Single-image inference
│       └── utils.py                # Seed, JSON I/O, directory helpers
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

---

## 🧬 Cell Classes

The model classifies blood cell images into the following 6 categories (2 classes excluded by default: `ig`, `neutrophil`):

| Class | Description |
|---|---|
| **Basophil** | White blood cell involved in allergic and inflammatory responses |
| **Eosinophil** | Linked with immune response and parasitic infections |
| **Erythroblast** | Immature red blood cell precursor |
| **Lymphocyte** | Critical for adaptive immune defense |
| **Monocyte** | Large WBC that differentiates into macrophages |
| **Platelet** | Cell fragment essential for blood clotting |

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/abhi-0404/blood-cell-cancer-classification.git
cd blood-cell-cancer-classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the package in editable mode (for local imports)

```bash
pip install -e .
```

---

## 📁 Dataset Layout

Place your dataset under `data/raw/` with one subfolder per class:

```text
data/raw/
├── basophil/
│   ├── img_001.jpg
│   └── img_002.jpg
├── eosinophil/
├── erythroblast/
├── lymphocyte/
├── monocyte/
└── platelet/
```

> You can also pass any custom path using `--data-dir`.

---

## 🚀 Training

### Train the custom CNN (baseline)

```bash
python scripts/train.py --data-dir data/raw --model cnn
```

### Train EfficientNetB3 (recommended)

```bash
python scripts/train.py --data-dir data/raw --model efficientnetb3
```

### Override hyperparameters at runtime

```bash
python scripts/train.py --data-dir data/raw --model efficientnetb3 --epochs 30 --batch-size 32
```

Training outputs saved to:
- `models/best_<model_name>.keras` — best checkpoint
- `models/class_indices.json` — label-to-index mapping
- `reports/<model_name>_metrics.json` — train/val/test accuracy & loss
- `reports/<model_name>_history.csv` — epoch-by-epoch history
- `reports/figures/<model_name>_confusion_matrix.png`

---

## 📊 Evaluation

```bash
python scripts/evaluate.py --data-dir data/raw --model-path models/best_efficientnetb3.keras
```

Add `--rescale` flag when evaluating a custom CNN model:

```bash
python scripts/evaluate.py --data-dir data/raw --model-path models/best_cnn.keras --rescale
```

---

## 🔍 Prediction

Predict a single image:

```bash
python scripts/predict.py --image path/to/image.jpg --model-path models/best_efficientnetb3.keras
```

Sample output:

```json
{
  "label": "eosinophil",
  "confidence": 0.9823,
  "probabilities": {
    "basophil": 0.0021,
    "eosinophil": 0.9823,
    "erythroblast": 0.0042,
    "lymphocyte": 0.0061,
    "monocyte": 0.0031,
    "platelet": 0.0022
  }
}
```

---

## 🖥️ Streamlit Interface

Launch the web app:

```bash
python -m streamlit run app.py
```

Open in browser: **http://localhost:8501**

### Interface Features

| Tab | Description |
|---|---|
| **Inference Workflow** | Upload a microscopy image and get instant classification |
| **Pipeline Architecture** | Visual overview of the 3-phase ML pipeline |
| **Model Metrics** | Displays train/val/test accuracy when reports are available |
| **Deployment** | Deployment options and instructions |

> **Demo Mode**: If no trained model is found, the app opens in demo mode — perfect for explaining the workflow without a trained model.

---

## 🏗️ Model Architecture

### Custom CNN

```
Input (224×224×3)
→ [Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)] × 1
→ [Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.30)] × 1
→ [Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.35)] × 1
→ GlobalAveragePooling2D
→ Dense(256, L2) → BN → Dropout(0.5)
→ Dense(num_classes, softmax)
```

### EfficientNetB3 (Transfer Learning)

```
Input (224×224×3)
→ EfficientNetB3 (ImageNet weights, frozen initially)
→ BatchNormalization
→ Dense(512, ReLU, L2)
→ Dropout(0.45)
→ Dense(num_classes, softmax)

Fine-tuning: Last 40 layers unfrozen, lr=1e-5
```

---

## 🔧 Configuration

All hyperparameters live in `configs/default.yaml`:

```yaml
image_size: [224, 224]
channels: 3
batch_size: 32
epochs: 20
fine_tune_epochs: 10
learning_rate: 0.001
fine_tune_learning_rate: 0.00001
train_split: 0.70
validation_split: 0.15
test_split: 0.15
random_state: 42
exclude_classes:
  - ig
  - neutrophil
model_dir: models
report_dir: reports
```

---

## ☁️ Deployment

### Streamlit Community Cloud (Recommended)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Deploy

> If the `.keras` model file exceeds GitHub's 100MB limit, use **Git LFS** or host the model on Hugging Face Hub / Google Drive and download it at startup.

### Git LFS for large model files

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
```

---

## 📓 Notebook

The original exploratory notebook is preserved at:

```
notebooks/blood-cell-cancer-using-cnn-and-efficientnetb3.ipynb
```

It contains the full EDA, model experiments, and training runs that informed the final package design.

---

## 🧪 Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| EfficientNetB3 | Transfer learning backbone |
| scikit-learn | Train/val/test splitting, metrics |
| Pandas | DataFrame-based data pipeline |
| Matplotlib / Seaborn | Confusion matrix visualization |
| Pillow / OpenCV | Image loading and preprocessing |
| Streamlit | Web interface |
| PyYAML | Config file parsing |

---

## 👤 Author

**Abhishek Kumar**
- GitHub: [@abhi-0404](https://github.com/abhi-0404)
- Email: kumarabhishek82929@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
