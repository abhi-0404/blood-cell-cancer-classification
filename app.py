from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from blood_cell_cancer.config import load_config


st.set_page_config(
    page_title="Blood Cell Cancer Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)


CLASS_DESCRIPTIONS = {
    "basophil": "White blood cell involved in allergic and inflammatory responses.",
    "eosinophil": "White blood cell linked with immune response and parasitic infections.",
    "erythroblast": "Immature red blood cell precursor.",
    "lymphocyte": "White blood cell important for immune defense.",
    "monocyte": "Large white blood cell that can become macrophages.",
    "platelet": "Cell fragment involved in blood clotting.",
}

# --- MASTER UI/UX ENHANCEMENT: ENTERPRISE SAAS THEME ---
st.markdown(
    """
    <style>
    /* Global Premium Dark Theme */
    .stApp {
        background-color: #050810;
        color: #E2E8F0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .block-container {
        padding-top: 2rem !important;
    }

    /* Sidebar Overhaul */
    [data-testid="stSidebar"] {
        background-color: #0A0F1C;
        border-right: 1px solid #1E293B;
    }
    
    h1, h2, h3 { color: #F8FAFC !important; font-weight: 600; }
    h1 { letter-spacing: -0.03em; font-size: 2rem; margin-bottom: 0.2rem;}
    .subtitle { color: #64748B; font-size: 1rem; margin-bottom: 2rem; border-bottom: 1px solid #1E293B; padding-bottom: 1rem;}
    
    .status-badge {
        display: flex;
        align-items: center;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 32px;
        background: #111827;
        border: 1px solid #1E293B;
    }
    .status-dot { width: 10px; height: 10px; border-radius: 50%; margin-right: 12px; }
    .dot-ready { background-color: #00E5FF; box-shadow: 0 0 12px rgba(0, 229, 255, 0.6); }
    .dot-missing { background-color: #F59E0B; box-shadow: 0 0 10px rgba(245, 158, 11, 0.5); }
    
    .sidebar-metric { margin-bottom: 20px; }
    .sidebar-metric-label { font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
    .sidebar-metric-value { font-size: 1.1rem; color: #E2E8F0; font-weight: 500; }

    .stTabs [data-baseweb="tab-list"] { gap: 40px; }
    .stTabs [data-baseweb="tab"] { color: #64748B; font-weight: 500; font-size: 1rem; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { color: #00E5FF !important; border-bottom: 2px solid #00E5FF !important; }

    /* ==============================================================
       MAIN PANELS: STRICT ADAPTIVE SYMMETRY (100% Equal Heights)
       ============================================================== */
       
    /* 1. Force columns to stretch equally */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    
    /* 2. Force internal column wrappers to take up full available height */
    [data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="column"] > div {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    
    /* 3. The actual visible box panel */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #0A0F1C !important;
        border: 1px solid #1E293B !important;
        border-radius: 16px !important;
        padding: 24px !important;
        flex-grow: 1 !important; /* Expands to fill the stretched column */
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    /* ==============================================================
       LEFT PANEL CONTROLS
       ============================================================== */
       
    [data-testid="stFileUploadDropzone"] {
        background-color: #0F1626 !important;
        border: 2px dashed #00E5FF !important;
        border-radius: 12px !important;
        padding: 40px 20px !important; 
        transition: all 0.3s ease;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 12px;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(0, 229, 255, 0.05) !important;
        box-shadow: inset 0 0 20px rgba(0, 229, 255, 0.1);
    }
    [data-testid="stFileUploadDropzone"] small {
        display: none !important;
    }

    /* New Inline Parameter Group (Replaces Expander) */
    .param-group {
        background: rgba(15, 22, 38, 0.6);
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 12px;
    }
    
    [data-testid="baseButton-primary"] {
        background-color: rgba(0, 229, 255, 0.1) !important;
        border: 1px solid #00E5FF !important;
        color: #00E5FF !important;
        border-radius: 8px !important;
        padding: 14px 24px !important;
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em;
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: auto !important; /* Pushes button to absolute bottom */
    }
    [data-testid="baseButton-primary"]:hover, [data-testid="baseButton-primary"]:active {
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.5) !important;
        background-color: #00E5FF !important;
        color: #050810 !important;
    }

    /* ==============================================================
       RIGHT PANEL: RESULTS & IDLE STATE
       ============================================================== */
       
    .results-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 16px;
        padding-bottom: 16px;
        border-bottom: 1px solid #1E293B;
    }

    .idle-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex-grow: 1; /* Pushes the footer down naturally */
        min-height: 250px;
        background: #0B101A;
        border: 1px solid #162032;
        border-radius: 12px;
        color: #64748B;
        text-align: center;
        margin-top: 8px;
        margin-bottom: 16px;
    }
    
    .result-card {
        background: rgba(0, 229, 255, 0.05);
        border: 1px solid rgba(0, 229, 255, 0.2);
        border-radius: 12px;
        padding: 32px 24px;
        margin-top: 16px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: inset 0 0 30px rgba(0, 229, 255, 0.05);
    }
    .result-class {
        font-size: 2.5rem; font-weight: 700; color: #00E5FF; margin: 0; letter-spacing: -0.02em; text-transform: uppercase;
    }
    
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
        background: #0B101A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
        margin-bottom: 12px;
        width: 100%;
    }
    [data-testid="stImage"] img {
        border-radius: 8px;
        max-height: 350px !important;
        width: auto !important;
        object-fit: contain !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_model_status(model_path: Path, class_indices_path: Path) -> tuple[bool, str]:
    if not model_path.exists():
        return False, "Model binary missing"
    if not class_indices_path.exists():
        return False, "Class indices missing"
    return True, "System Operational"


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image, image_size: tuple[int, int], rescale: bool) -> np.ndarray:
    image = image.convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    if rescale:
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div style="border: 1px solid #1E293B; background: #0A0F1C; border-radius: 12px; padding: 24px; height: 100%;">
            <h4 style="color: #F8FAFC; margin-bottom: 12px; font-size: 1.1rem;">{title}</h4>
            <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5; margin: 0;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar_metric(label: str, value: str):
    st.markdown(
        f"""
        <div class="sidebar-metric">
            <div class="sidebar-metric-label">{label}</div>
            <div class="sidebar-metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True
    )


# --- BACKEND LOGIC ---
config = load_config(ROOT / "configs" / "default.yaml")
model_path = ROOT / "models" / "best_efficientnetb3.keras"
class_indices_path = ROOT / "models" / "class_indices.json"
metrics_path = ROOT / "reports" / "efficientnetb3_metrics.json"
confusion_matrix_path = ROOT / "reports" / "figures" / "efficientnetb3_confusion_matrix.png"

class_indices = load_json(class_indices_path)
if class_indices:
    classes = [label for label, _ in sorted(class_indices.items(), key=lambda item: item[1])]
else:
    classes = ["basophil", "eosinophil", "erythroblast", "lymphocyte", "monocyte", "platelet"]

model_ready, model_message = get_model_status(model_path, class_indices_path)

if "analyze_triggered" not in st.session_state:
    st.session_state.analyze_triggered = False

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### System Diagnostics")
    st.write("") 
    status_dot = "dot-ready" if model_ready else "dot-missing"
    status_text = "ENGINE ONLINE" if model_ready else "DEMO MODE"
    st.markdown(
        f"""
        <div class="status-badge">
            <div class="status-dot {status_dot}"></div>
            {status_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    render_sidebar_metric("Target Architecture", "EfficientNetB3")
    render_sidebar_metric("Input Resolution", f"{config.image_size[0]} x {config.image_size[1]}px")
    render_sidebar_metric("Classification Matrix", f"{len(classes)} Active Classes")
    render_sidebar_metric("System Status", model_message)
    
    if not model_ready:
        st.info("Presentation mode active. Connect compiled model to enable inference.")

# --- MAIN CANVAS ---
st.markdown("<h1>Blood Cell Classification Engine</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a microscopy sample to initiate deep learning analysis.</div>", unsafe_allow_html=True)

tab_predict, tab_pipeline, tab_results, tab_deploy = st.tabs(
    ["Inference Workflow", "Pipeline Architecture", "Model Metrics", "Deployment"]
)

with tab_predict:
    st.write("") 
    left_col, right_col = st.columns([1, 1], gap="large")
    
    with left_col:
        with st.container(border=True):
            
            uploaded_file = st.file_uploader(
                "Select or drop microscopy sample (JPG, PNG, BMP)", 
                type=["jpg", "jpeg", "png", "bmp"], 
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            else:
                image = None
                st.session_state.analyze_triggered = False
            
            # --- REMOVED EXPANDER: Parameters are now explicitly shown in a custom inline box ---
            st.markdown("<div class='param-group'>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                is_blood_cell_image = st.checkbox("Verify domain", value=True)
            with col2:
                rescale = st.checkbox("CNN rescale", value=False)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Action Button
            analyze_button = st.button("Analyze Blood Sample", type="primary", use_container_width=True)
            if analyze_button and image is not None:
                st.session_state.analyze_triggered = True

    with right_col:
        with st.container(border=True):
            st.markdown("<div class='results-header'>Diagnostic Results</div>", unsafe_allow_html=True)
            
            if image is None or not st.session_state.analyze_triggered:
                st.markdown(
                    """
                    <div class="idle-state">
                        <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#334155" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom:16px;">
                            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                            <line x1="12" y1="22.08" x2="12" y2="12"></line>
                        </svg>
                        <div style="font-weight: 500; font-size: 1.1rem; color: #94A3B8;">Awaiting sample input.</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    """
                    <div style="margin-top: auto; border-top: 1px solid #1E293B; padding-top: 16px;">
                        <div style="color: #64748B; font-size: 0.85rem; font-family: monospace;">Model: <span style="color: #00E5FF;">best_model.keras</span></div>
                        <div style="color: #64748B; font-size: 0.85rem; font-family: monospace; margin-top: 4px;">Classes: <span style="color: #00E5FF;">class_indices.json</span></div>
                    </div>
                    """, unsafe_allow_html=True
                )

            elif model_ready and st.session_state.analyze_triggered:
                if not is_blood_cell_image:
                    st.error("Inference halted. Domain verification required.")
                else:
                    with st.spinner("Executing neural analysis..."):
                        model = load_trained_model(str(model_path))
                        batch = preprocess_image(image, config.image_size, rescale=rescale)
                        probabilities = model.predict(batch, verbose=0)[0]
                        result = {classes[index]: float(probabilities[index]) for index in range(len(classes))}
                        predicted_label = max(result, key=result.get)
                        confidence = result[predicted_label] * 100
                    
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <p style="color: #64748B; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Primary Classification</p>
                            <h2 class="result-class">{predicted_label}</h2>
                            <p style="color: #94A3B8; margin-top: 12px; font-size: 1.1rem;">Confidence Index: <strong style="color: #F8FAFC;">{confidence:.1f}%</strong></p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("**Probability Distribution**", unsafe_allow_html=True)
                    st.bar_chart(pd.DataFrame({"Probability": result}), height=150)
                    
                    with st.expander("Clinical Pathology Note"):
                        st.write(CLASS_DESCRIPTIONS.get(predicted_label, "No specific pathology description available for this class."))
            else:
                st.warning("Inference engine offline. Train the target model to proceed.")

with tab_pipeline:
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1: card("Phase 1: Ingestion", "Load image directories, scale raw pixel inputs, and split sets (Train/Val/Test).")
    with c2: card("Phase 2: Extraction", "Compute deep hierarchical features via EfficientNetB3 transfer learning.")
    with c3: card("Phase 3: Validation", "Generate confusion matrices and calculate strict precision/recall metrics.")
    st.write("")
    st.code("python scripts/train.py --data-dir data/raw --model efficientnetb3", language="bash")

with tab_results:
    st.write("")
    metrics = load_json(metrics_path)
    if metrics:
        col1, col2, col3 = st.columns(3)
        with col1: card("Training Accuracy", f"{metrics.get('train', {}).get('accuracy', 0) * 100:.1f}%")
        with col2: card("Validation Accuracy", f"{metrics.get('validation', {}).get('accuracy', 0) * 100:.1f}%")
        with col3: card("Holdout Test Accuracy", f"{metrics.get('test', {}).get('accuracy', 0) * 100:.1f}%")
    else:
        st.info("Evaluation telemetry unavailable. Run training pipeline to populate metrics.")

with tab_deploy:
    st.write("")
    d1, d2, d3 = st.columns(3)
    with d1: card("Local Host", "Execute Streamlit natively for rapid prototyping and local diagnostics.")
    with d2: card("Cloud Edge", "Push to GitHub and deploy continuously via Streamlit Community Cloud.")
    with d3: card("Artifacts", "Store heavy `.keras` compiled binaries efficiently using Git Large File Storage (LFS).")