"""
Modern analytical Streamlit dashboard for Solar Panel Fault Detection.
Modularized and Refactored for clean architecture.
"""

import os
import sys
import io
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from PIL import Image

# =============================================================================
# Path & Imports Configuration
# =============================================================================
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.utils import load_config
from src.predict import Predictor

# =============================================================================
# Page Configuration & CSS
# =============================================================================
def init_page_config():
    st.set_page_config(
        page_title="Solar Panel Fault Analytics",
        page_icon="☀️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def inject_custom_css():
    """Encapsulates all custom styling to keep the main flow clean."""
    st.markdown(
        """
        <style>
            :root {
                --bg-1: #08111f; --bg-2: #0b1324;
                --card: rgba(17, 26, 46, 0.88); --card-2: rgba(15, 23, 42, 0.92);
                --border: rgba(148, 163, 184, 0.18);
                --text: #e5eefc; --muted: #8ea0bc;
            }
            .stApp {
                background: radial-gradient(circle at top left, rgba(59,130,246,0.10), transparent 28%),
                            radial-gradient(circle at top right, rgba(14,165,233,0.10), transparent 22%),
                            linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
                color: var(--text);
            }
            .hero {
                background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(15,23,42,0.78));
                border: 1px solid var(--border); border-radius: 24px;
                padding: 1.25rem 1.4rem; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.20);
                margin-bottom: 1rem;
            }
            .hero h1 { margin: 0; color: white; font-size: 2rem; line-height: 1.2; }
            .hero p { margin: 0.35rem 0 0 0; color: var(--muted); font-size: 0.95rem; }
            .section-card {
                background: var(--card); border: 1px solid var(--border);
                border-radius: 22px; padding: 1rem;
                box-shadow: 0 18px 36px rgba(0,0,0,0.16); margin-bottom: 1rem;
            }
            .pill {
                display: inline-block; padding: 0.28rem 0.7rem; border-radius: 999px;
                background: rgba(59,130,246,0.16); border: 1px solid rgba(59,130,246,0.30);
                color: #dbeafe; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.5rem;
            }
            .soft-card {
                background: linear-gradient(180deg, rgba(59,130,246,0.11), rgba(15,23,42,0.90));
                border: 1px solid var(--border); border-radius: 18px; padding: 0.85rem 0.9rem;
            }
            .subtle { color: var(--muted); font-size: 0.88rem; }
            .stTabs [data-baseweb="tab"] {
                background: rgba(15, 23, 42, 0.90); border-radius: 14px 14px 0 0;
                border: 1px solid var(--border); padding: 0.58rem 1rem;
            }
            .stTabs [aria-selected="true"] { background: rgba(59,130,246,0.16) !important; color: white !important; }
            div[data-testid="stMetric"] {
                background: rgba(15, 23, 42, 0.74); border: 1px solid var(--border);
                border-radius: 18px; padding: 0.65rem 0.85rem;
            }
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #07101d 0%, #0b1324 100%);
                border-right: 1px solid var(--border);
            }
            .warning-box { background: rgba(245, 158, 11, 0.10); border: 1px solid rgba(245, 158, 11, 0.25); padding: 0.85rem 1rem; border-radius: 14px; color: #fde68a; margin-top: 0.5rem; }
            .success-box { background: rgba(34, 197, 94, 0.10); border: 1px solid rgba(34, 197, 94, 0.25); padding: 0.85rem 1rem; border-radius: 14px; color: #bbf7d0; margin-top: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Core State & ML Helpers
# =============================================================================
def init_session_state():
    if "pred_history" not in st.session_state:
        st.session_state["pred_history"] = []

@st.cache_data
def load_app_config() -> Dict:
    return load_config()

def resolve_weights_path() -> Optional[str]:
    candidates = ["models/checkpoints/best_model.pth", "models/checkpoints/checkpoint_epoch_35.pth"]
    return next((path for path in candidates if os.path.exists(path)), None)

@st.cache_resource
def get_predictor() -> Predictor:
    return Predictor(load_app_config(), weights_path=resolve_weights_path())

# =============================================================================
# Utility Functions (File/Data Processing)
# =============================================================================
def handle_temp_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix if getattr(uploaded_file, "name", None) else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def analyze_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    if not probs:
        return {"top1": 0.0, "top2": 0.0, "margin": 0.0, "entropy": 0.0}
    sorted_probs = sorted(probs.values(), reverse=True)
    top1 = float(sorted_probs[0]) if len(sorted_probs) > 0 else 0.0
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    
    arr = np.clip(np.array(sorted_probs, dtype=np.float64), 1e-12, 1.0)
    entropy = float(-np.sum(arr * np.log(arr)))
    
    return {"top1": top1, "top2": top2, "margin": top1 - top2, "entropy": entropy}

# =============================================================================
# Component Renderers
# =============================================================================
def render_sidebar():
    st.sidebar.markdown("## ⚙️ Control Panel")
    weights = resolve_weights_path()
    st.sidebar.markdown(
        f"""
        <div class="soft-card">
            <div class="pill">ResNet50 only</div>
            <div style="color: white; font-weight: 700; font-size: 1rem;">Solar Fault Analytics</div>
            <div class="subtle" style="margin-top:0.2rem;">{weights if weights else "No checkpoint found"}</div>
        </div>
        """, unsafe_allow_html=True
    )
    
    controls = {
        "threshold": st.sidebar.slider("Display confidence threshold", 0.00, 1.00, 0.50, 0.01),
        "top_k": st.sidebar.slider("Top-K predictions", 1, 4, 4),
        "show_technical": st.sidebar.checkbox("Show technical details", value=True)
    }

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear session history"):
        st.session_state["pred_history"] = []
        st.rerun()

    st.sidebar.caption("Focus: Classification. YOLO has been removed from the UI and inference flow.")
    return controls

def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="pill">Solar Panel Fault Detection</div>
            <h1>Modern Analytical Dashboard</h1>
            <p>Clean, product-style interface for ResNet50 inference, confidence analysis, and session insights.</p>
        </div>
        """, unsafe_allow_html=True
    )

# --- Tab 1: Single Prediction ---
def render_single_prediction_tab(predictor: Predictor, controls: Dict):
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Upload a solar panel image")
        
        uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"], key="single_upload", label_visibility="collapsed")
        
        if uploaded:
            # FIX: Replaced use_container_width=True with a fixed width to prevent massive images
            st.image(Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB"), caption="Input image", width=450)
            temp_path = handle_temp_file(uploaded)

            with st.spinner("Running inference..."):
                t0 = time.perf_counter()
                try:
                    result = predictor.predict_single(temp_path)
                    inference_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                    result, inference_ms = None, 0.0
            
            if os.path.exists(temp_path): os.remove(temp_path)

            if result:
                process_and_display_single_result(result, inference_ms, controls, uploaded.name)
        else:
            st.info("Upload a solar panel image to begin.")
            
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        render_recent_history_sidebar()

def process_and_display_single_result(result: Dict, inference_ms: float, controls: Dict, filename: str):
    probs = result.get("all_probabilities", {})
    analysis = analyze_probabilities(probs)
    conf = float(result.get("confidence", 0.0))

    # Render KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted class", str(result.get("fault_type", "unknown")).upper())
    c2.metric("Confidence", f"{conf:.2%}")
    c3.metric("Latency", f"{inference_ms:.0f} ms")
    c4.metric("Margin / Entropy", f"{analysis['margin']:.2%} | {analysis['entropy']:.2f}")

    # Alert Box
    if conf < controls["threshold"]:
        st.markdown('<div class="warning-box">Confidence below display threshold. Model is uncertain.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">Confident prediction ready for review.</div>', unsafe_allow_html=True)

    # Technical Expander
    if controls["show_technical"]:
        with st.expander("Technical details", expanded=False):
            st.json(probs)
            
    # Update State
    st.session_state["pred_history"].insert(0, {
        "time": pd.Timestamp.now(), "filename": filename,
        "fault_type": result.get("fault_type", "unknown"),
        "confidence": conf, "margin": analysis["margin"],
        "entropy": analysis["entropy"], "latency_ms": inference_ms,
    })
    st.session_state["pred_history"] = st.session_state["pred_history"][:50]

def render_recent_history_sidebar():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Decision view")
    
    if st.session_state["pred_history"]:
        latest = st.session_state["pred_history"][0]
        st.metric("Latest prediction", str(latest["fault_type"]).upper())
        
        hist_df = pd.DataFrame(st.session_state["pred_history"])
        hist_df["time"] = pd.to_datetime(hist_df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(hist_df[["time", "filename", "fault_type", "confidence"]], use_container_width=True, hide_index=True)
    else:
        st.info("Your recent predictions will appear here.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 2: Batch Analysis ---
def render_batch_analysis_tab(predictor: Predictor):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Batch image analysis")
    
    batch_files = st.file_uploader("Choose multiple images", type=["jpg", "png", "bmp"], accept_multiple_files=True, key="batch", label_visibility="collapsed")
    
    if batch_files and st.button("Run batch inference"):
        results = []
        bar = st.progress(0)
        
        for i, f in enumerate(batch_files):
            temp_path = handle_temp_file(f)
            t0 = time.perf_counter()
            try:
                res = predictor.predict_single(temp_path)
                analysis = analyze_probabilities(res.get("all_probabilities", {}))
                results.append({
                    "filename": f.name, "fault_type": res.get("fault_type", "unknown"),
                    "confidence": float(res.get("confidence", 0.0)),
                    "latency_ms": (time.perf_counter() - t0) * 1000.0
                })
            except Exception as e:
                results.append({"filename": f.name, "fault_type": "error", "confidence": 0.0, "latency_ms": 0.0})
            
            if os.path.exists(temp_path): os.remove(temp_path)
            bar.progress((i + 1) / len(batch_files))
            
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 3: Insights ---
def render_insights_tab():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Session analytics")
    
    if st.session_state["pred_history"]:
        df = pd.DataFrame(st.session_state["pred_history"])
        c1, c2 = st.columns(2)
        c1.metric("Total Predictions", len(df))
        c2.metric("Avg Confidence", f"{df['confidence'].mean():.2%}")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Make a prediction to unlock analytics.")
        
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# Main Execution Layout
# =============================================================================
def main():
    init_page_config()
    inject_custom_css()
    init_session_state()

    # Load resources
    predictor = get_predictor()
    
    # Render UI
    controls = render_sidebar()
    render_hero()

    tab_single, tab_batch, tab_insights = st.tabs(["🔍 Single Prediction", "📦 Batch Analysis", "📈 Insights & Artifacts"])

    with tab_single:
        render_single_prediction_tab(predictor, controls)
        
    with tab_batch:
        render_batch_analysis_tab(predictor)
        
    with tab_insights:
        render_insights_tab()

    st.markdown("---")
    st.caption("Solar Panel Fault Detection • ResNet50-dashboard • Clean analytics UI")

if __name__ == "__main__":
    main()
