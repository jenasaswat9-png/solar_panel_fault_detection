"""
Modern Streamlit Dashboard for Solar Panel Fault Detection
- Drop-in replacement for dashboard/streamlit_app.py
- Cross-platform temporary file handling
- Interactive charts (Plotly) + training stats if present
"""

import os
import sys
from pathlib import Path
import tempfile
import io
import json
import time
from typing import Dict, List, Optional

import numpy as np
import cv2
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

# add parent project to path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.utils import load_config, get_device
from src.predict import Predictor

# ----------------------------
# Page + CSS
# ----------------------------
st.set_page_config(page_title="Solar Panel Fault Detection",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   page_icon="☀️")

st.markdown("""
<style>
/* Modern look */
body { background-color: #0f1724; color: #e6eef8; }
.reportview-container .main .block-container{ padding-top:1rem; }
.stButton>button { background-color: #2563eb; color: white; }
.metric-card {
    background: linear-gradient(90deg, rgba(37,99,235,0.12), rgba(2,6,23,0.0));
    border-radius: 12px; padding: 12px;
    box-shadow: 0 8px 20px rgba(2,6,23,0.6);
}
.small-muted { color: #9aa7bf; font-size:12px; }
.header { font-size:28px; font-weight:600; color: #fff; }
.sidebar .stSlider>div>div>div>input { color: #fff; }
.card { background: #071029; border-radius:12px; padding:12px; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Helpers
# ----------------------------

@st.cache_resource
def get_predictor(model_type: str = "resnet50") -> Predictor:
    cfg = load_config()
    # default weights path used by the repo
    weights_path = "models/checkpoints/best_model.pth" if model_type == "resnet50" else None
    if weights_path and not os.path.exists(weights_path):
        weights_path = None
    return Predictor(cfg, model_type=model_type, weights_path=weights_path)

def save_temp_image_from_upload(uploaded_file) -> str:
    """Write uploaded file to a safe temp file and return path (cross-platform)."""
    image = Image.open(uploaded_file)
    arr = np.array(image)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    path = tmp.name
    tmp.close()
    cv2.imwrite(path, arr)
    return path

def save_temp_image_array(image_array: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    path = tmp.name
    tmp.close()
    cv2.imwrite(path, image_array)
    return path

def draw_boxes_on_image(image_bgr: np.ndarray, detections: List[Dict], show_labels=True) -> np.ndarray:
    img = image_bgr.copy()
    color_map = {
        "normal": (46,204,113),
        "cracked": (231,76,60),
        "hotspot": (230,126,34),
        "dust": (241,196,15)
    }
    for det in detections:
        bbox = det.get("bbox", [0,0,0,0])
        cl = det.get("class", "unk")
        conf = det.get("confidence", 0.0)
        x1,y1,x2,y2 = map(int, bbox)
        color = color_map.get(cl, (255,255,255))
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        if show_labels:
            label = f"{cl}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

def load_json_if_exists(path) -> Optional[Dict]:
    path = Path(path)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def probs_to_df(probs: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"class": list(probs.keys()), "prob": list(probs.values())})

def results_to_csv(results: List[Dict]) -> str:
    df = pd.DataFrame(results)
    return df.to_csv(index=False)

# ----------------------------
# Sidebar - Controls
# ----------------------------
st.sidebar.markdown("## ⚙️ Model & Settings")
model_type = st.sidebar.selectbox("Model", ["resnet50", "yolo"], index=0, help="Select model to use")
predictor = get_predictor(model_type)
load_msg = f"✅ {model_type.upper()} loaded" if predictor else "⚠️ No model weights found, predictor created"
st.sidebar.success(load_msg)

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
show_boxes = st.sidebar.checkbox("Show bounding boxes (if any)", True)
top_k = st.sidebar.slider("Top-K probabilities to show", 1, 4, 4)

st.sidebar.markdown("---")
st.sidebar.markdown("### Demo / Quick actions")
if st.sidebar.button("Load random sample from dataset"):
    # provide a quick sample path (if dataset present)
    sample_base = Path("data/processed/test")
    sample_img = None
    if sample_base.exists():
        # pick class with most images
        all_imgs = list(sample_base.rglob("*.jpg"))
        if all_imgs:
            sample_img = str(all_imgs[0])
    if sample_img:
        st.session_state["_demo_sample"] = sample_img
        st.experimental_rerun()
    else:
        st.sidebar.warning("No dataset samples found in data/processed/test")

st.sidebar.markdown("---")
st.sidebar.markdown("### Project stats & artifacts")
if st.sidebar.button("Open results folder"):
    st.sidebar.info("Open the 'results' and 'models' folder in VS Code / File Explorer to inspect artifacts.")

# ----------------------------
# Main layout
# ----------------------------
st.markdown("<div class='header'>☀️ Solar Panel Fault Detection — Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Upload an image (or multiple) to get predictions. Uses ResNet50 classification or YOLO detection (if available).</div>", unsafe_allow_html=True)
st.write("")

col_left, col_right = st.columns((2, 3))

# Left column: upload & single predict
with col_left:
    st.markdown("### 📤 Single Image Inference", unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag & drop an image or click to browse", type=["jpg","jpeg","png"], accept_multiple_files=False, key="single_uploader")

    # allow demo variable (from sidebar button)
    if "_demo_sample" in st.session_state and not uploaded:
        # attempt to populate from path
        try:
            demo_path = st.session_state["_demo_sample"]
            uploaded = open(demo_path, "rb")
        except Exception:
            uploaded = None

    if uploaded:
        # show preview
        try:
            thumb = Image.open(uploaded)
        except Exception:
            uploaded.seek(0)
            thumb = Image.open(uploaded)
        st.image(thumb, caption="Uploaded image preview", use_column_width=True)

        # Save temp file cross-platform and call predictor
        tmp_path = save_temp_image_from_upload(uploaded)
        with st.spinner("🔎 Running model..."):
            try:
                result = predictor.predict_single(tmp_path)
            except Exception as e:
                st.error(f"Model error: {e}")
                result = None
        # cleanup
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if result:
            # summary metrics
            st.markdown("#### ⚡ Prediction summary")
            fault = result.get("fault_type", "unknown")
            conf = float(result.get("confidence", 0.0))
            inf_time = float(result.get("inference_time", 0.0)) * 1000
            probs = result.get("all_probabilities", {})

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Fault", str(fault).upper())
            metric_col2.metric("Confidence", f"{conf:.2%}")
            metric_col3.metric("Latency", f"{inf_time:.0f} ms")

            # show probability bar (plotly)
            if probs:
                dfp = probs_to_df(probs).sort_values("prob", ascending=True)
                fig = go.Figure(go.Bar(
                    x=dfp["prob"],
                    y=dfp["class"],
                    orientation="h",
                    marker=dict(color=dfp["prob"], colorscale="Blues"),
                    hovertemplate="%{y}: %{x:.2%}<extra></extra>"
                ))
                fig.update_layout(height=240, margin=dict(l=10,r=10,t=20,b=20), xaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)

            # show detection visualization (if any)
            st.markdown("---")
            st.markdown("#### 🔍 Visualization")
            # if detections exist, draw boxes; else show raw
            original_arr = np.array(thumb)
            if len(original_arr.shape) == 3 and original_arr.shape[2] == 3:
                original_bgr = cv2.cvtColor(original_arr, cv2.COLOR_RGB2BGR)
            else:
                original_bgr = original_arr
            viz_arr = original_bgr
            if show_boxes and result.get("detections"):
                viz_arr = draw_boxes_on_image(original_bgr, result.get("detections", []), show_labels=True)
            viz_rgb = cv2.cvtColor(viz_arr, cv2.COLOR_BGR2RGB) if len(viz_arr.shape)==3 else viz_arr
            st.image(viz_rgb, use_column_width=True)

            # detections table
            if result.get("detections"):
                st.markdown("**Detected objects**")
                dets = result["detections"]
                df = []
                for i, d in enumerate(dets):
                    df.append({
                        "id": i+1,
                        "class": d.get("class"),
                        "confidence": f"{d.get('confidence',0.0):.2%}",
                        "bbox": f"{tuple(map(lambda x: round(x,1), d.get('bbox',[])))}"
                    })
                st.table(pd.DataFrame(df))

            # record into session history
            history = st.session_state.get("pred_history", [])
            hist_entry = {
                "timestamp": time.time(),
                "filename": getattr(uploaded, "name", "uploaded_image"),
                "fault_type": fault,
                "confidence": conf
            }
            history.insert(0, hist_entry)
            st.session_state["pred_history"] = history[:50]  # keep recent 50

            # save visualization to a downloadable file
            buf = cv2.imencode(".jpg", cv2.cvtColor(viz_rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
            st.download_button("📥 Download visualization", data=buf, file_name="prediction_viz.jpg", mime="image/jpeg")
        else:
            st.warning("Prediction failed. Check model logs in console.")

    # quick session history view
    st.markdown("---")
    st.markdown("#### 🕘 Recent predictions (session)")
    hist = st.session_state.get("pred_history", [])
    if hist:
        hist_df = pd.DataFrame(hist)
        hist_df["time"] = pd.to_datetime(hist_df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        st.table(hist_df[["time","filename","fault_type","confidence"]].head(8))
        st.download_button("📥 Download session CSV", data=hist_df.to_csv(index=False), file_name="session_predictions.csv")
    else:
        st.write("No predictions yet in this session.")

# Right column: batch, stats, training curves
with col_right:
    st.markdown("### 📊 Batch Analysis & Model Stats", unsafe_allow_html=True)

    # Batch uploader
    st.markdown("#### Batch image analysis")
    batch_files = st.file_uploader("Upload multiple images (max 40)", type=["jpg","jpeg","png"], accept_multiple_files=True, key="batch_uploader")
    if batch_files:
        max_files = 40
        files = batch_files[:max_files]
        progress = st.progress(0)
        results = []
        for i, f in enumerate(files):
            tmp_path = save_temp_image_from_upload(f)
            try:
                res = predictor.predict_single(tmp_path)
            except Exception as e:
                res = {"fault_type": "error", "confidence": 0.0, "error": str(e)}
            res["filename"] = getattr(f, "name", f"file_{i}")
            results.append(res)
            try: os.remove(tmp_path)
            except Exception: pass
            progress.progress((i+1)/len(files))
        # summarize
        df = pd.DataFrame([{"filename": r["filename"], "fault_type": r.get("fault_type",""), "confidence": r.get("confidence",0.0)} for r in results])
        st.markdown("**Batch results**")
        st.dataframe(df)
        # pie chart
        counts = df["fault_type"].value_counts().reset_index()
        counts.columns = ["fault","count"]
        figp = px.pie(counts, values="count", names="fault", title="Fault distribution", color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(figp, use_container_width=True)
        # download results
        st.download_button("📥 Download batch CSV", data=df.to_csv(index=False), file_name="batch_results.csv", mime="text/csv")

    # Model statistics (from results if available)
    st.markdown("#### Model performance & artifacts")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    training_history = load_json_if_exists(BASE_DIR / "results" / "training_history.json")
    conf_mat_exists = (BASE_DIR / "results" / "confusion_matrix.png").exists()
    if training_history:
        last = training_history.get("history", {})
        # attempt to show last saved best metrics or metrics in training_history
        # fallback to placeholders
        val_acc = training_history.get("best_val_acc") or training_history.get("best_accuracy") or training_history.get("best_val_accuracy") or 0.0
        metric_col1.metric("Best Val Accuracy", f"{val_acc:.2%}" if isinstance(val_acc,(float,int)) else str(val_acc), delta=None)
        metric_col2.metric("Precision (est)", training_history.get("precision", "N/A"))
        metric_col3.metric("Recall (est)", training_history.get("recall", "N/A"))
        metric_col4.metric("mAP (est)", training_history.get("mAP", "N/A"))
    else:
        metric_col1.metric("Best Val Accuracy", "—")
        metric_col2.metric("Precision", "—")
        metric_col3.metric("Recall", "—")
        metric_col4.metric("mAP", "—")

    # show confusion matrix image if exists
    if conf_mat_exists:
        st.markdown("**Confusion matrix (test set)**")
        st.image(str(BASE_DIR / "results" / "confusion_matrix.png"), use_column_width=True)
    else:
        st.info("Confusion matrix not found in results/confusion_matrix.png (optional).")

    # training curves
    st.markdown("#### Training curves")
    if training_history:
        try:
            # many possible formats — try common structure
            hist = training_history
            epochs = range(1, len(hist.get("train_loss", hist.get("loss", []))) + 1)
            train_loss = hist.get("train_loss") or hist.get("loss") or []
            val_loss = hist.get("val_loss") or hist.get("validation_loss") or []
            fig = go.Figure()
            if train_loss:
                fig.add_trace(go.Scatter(x=list(epochs)[:len(train_loss)], y=train_loss, mode="lines+markers", name="Train Loss"))
            if val_loss:
                fig.add_trace(go.Scatter(x=list(epochs)[:len(val_loss)], y=val_loss, mode="lines+markers", name="Val Loss"))
            fig.update_layout(title="Training / Validation Loss", xaxis_title="Epoch", yaxis_title="Loss", height=320)
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Training history exists but could not render curves (unexpected format).")
    else:
        st.info("Training history not found at results/training_history.json")

    st.markdown("---")
    st.markdown("#### Quick demo controls")
    st.write("Use the buttons below to test locally saved images (use correct paths).")
    demo_path = st.text_input("Path to local image (optional)", value="")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        if st.button("Run demo image"):
            if demo_path and os.path.exists(demo_path):
                tmp = save_temp_image_array(cv2.imread(demo_path))
                r = predictor.predict_single(tmp)
                os.remove(tmp)
                st.success(f"Prediction: {r.get('fault_type')} ({r.get('confidence'):.2%})")
            else:
                st.warning("Provide valid local path or upload a file above.")

    with col_d2:
        if st.button("Clear session history"):
            st.session_state["pred_history"] = []
            st.experimental_rerun()

# footer
st.markdown("---")
st.markdown("<div class='small-muted'>Tip: rename files to remove spaces/special characters to avoid path issues. For production, run the FastAPI server for real-time API-based inference.</div>", unsafe_allow_html=True)

