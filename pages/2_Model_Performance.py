"""
Authentica AI — Dynamic Model Performance & Scientific Evaluation Dashboard.
"""
import json
import subprocess
import sys
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Model Performance — Authentica AI",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"

st.title("📊 Scientific Model Performance & Evaluation")
st.markdown(
    """
    This dashboard displays **real, dynamically computed evaluation metrics** from our held-out 
    test datasets. In accordance with our **Zero-Fabrication Policy**, all figures, confusion matrices, 
    and ROC curves are generated directly by executing evaluation scripts over labeled test samples.
    """
)

tab_img_perf, tab_txt_perf, tab_eval_mgmt = st.tabs([
    "🖼️ Image Detector Benchmark",
    "📝 Text Detector Benchmark",
    "⚙️ Benchmark Execution & Management",
])


def load_metric_report(modality: str):
    """Safely loads evaluated metric JSON payload."""
    metric_file = METRICS_DIR / f"{modality}_metrics.json"
    if not metric_file.exists():
        return None
    try:
        with open(metric_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# TAB 1: IMAGE DETECTOR BENCHMARK
# -----------------------------------------------------------------------------
with tab_img_perf:
    st.subheader("🖼️ Image Detection Benchmark Evaluation")
    img_report = load_metric_report("image")

    if img_report is None:
        st.warning("⚠️ No image evaluation results available yet. Run the benchmark to generate metrics.")
    else:
        m = img_report["metrics"]
        st.caption(f"**Model ID:** `{img_report.get('model_name')}` | **Dataset:** `{img_report.get('metadata', {}).get('dataset_name', 'Sanity Test Set')}`")

        # Top-level Metric Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sample Count", m["sample_count"])
        c2.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
        c3.metric("Precision", f"{m['precision'] * 100:.1f}%")
        c4.metric("Recall", f"{m['recall'] * 100:.1f}%")
        c5.metric("F1-Score", f"{m['f1_score'] * 100:.1f}%")

        c_auc, c_spec, c_thresh, _ = st.columns(4)
        c_auc.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "N/A")
        c_spec.metric("Specificity", f"{m['specificity'] * 100:.1f}%")
        c_thresh.metric("Operating Threshold", f"{m['threshold']:.2f}")

        st.markdown("---")
        st.markdown("### 📈 Visual Benchmark Artifacts")

        col_cm, col_roc = st.columns(2)

        cm_fig_path = FIGURES_DIR / "image_confusion_matrix.png"
        if cm_fig_path.exists():
            with col_cm:
                st.image(str(cm_fig_path), caption="Image Detector Confusion Matrix", use_container_width=True)

        roc_fig_path = FIGURES_DIR / "image_roc_curve.png"
        if roc_fig_path.exists():
            with col_roc:
                st.image(str(roc_fig_path), caption="Image Detector ROC Curve", use_container_width=True)

        dist_fig_path = FIGURES_DIR / "image_score_distribution.png"
        if dist_fig_path.exists():
            st.image(str(dist_fig_path), caption="AI Detection Score Distribution (Human vs AI)", use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: TEXT DETECTOR BENCHMARK
# -----------------------------------------------------------------------------
with tab_txt_perf:
    st.subheader("📝 Text Detection Benchmark Evaluation")
    txt_report = load_metric_report("text")

    if txt_report is None:
        st.warning("⚠️ No text evaluation results available yet. Run the benchmark to generate metrics.")
    else:
        m = txt_report["metrics"]
        st.caption(f"**Model ID:** `{txt_report.get('model_name')}` | **Dataset:** `{txt_report.get('metadata', {}).get('dataset_name', 'HC3 Sanity Benchmark')}`")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Sample Count", m["sample_count"])
        c2.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
        c3.metric("Precision", f"{m['precision'] * 100:.1f}%")
        c4.metric("Recall", f"{m['recall'] * 100:.1f}%")
        c5.metric("F1-Score", f"{m['f1_score'] * 100:.1f}%")

        c_auc, c_spec, c_thresh, _ = st.columns(4)
        c_auc.metric("ROC-AUC", f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "N/A")
        c_spec.metric("Specificity", f"{m['specificity'] * 100:.1f}%")
        c_thresh.metric("Operating Threshold", f"{m['threshold']:.2f}")

        st.markdown("---")
        st.markdown("### 📈 Visual Benchmark Artifacts")

        col_cm, col_roc = st.columns(2)

        cm_fig_path = FIGURES_DIR / "text_confusion_matrix.png"
        if cm_fig_path.exists():
            with col_cm:
                st.image(str(cm_fig_path), caption="Text Detector Confusion Matrix", use_container_width=True)

        roc_fig_path = FIGURES_DIR / "text_roc_curve.png"
        if roc_fig_path.exists():
            with col_roc:
                st.image(str(roc_fig_path), caption="Text Detector ROC Curve", use_container_width=True)

        dist_fig_path = FIGURES_DIR / "text_score_distribution.png"
        if dist_fig_path.exists():
            st.image(str(dist_fig_path), caption="AI Detection Score Distribution (Human vs AI)", use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: BENCHMARK MANAGEMENT & LIVE RUNNER
# -----------------------------------------------------------------------------
with tab_eval_mgmt:
    st.subheader("⚙️ Execute Labeled Benchmarks")
    st.write("Trigger offline evaluation scripts directly to regenerate metrics and figures from held-out test data.")

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("#### 🖼️ Run Image Evaluation")
        if st.button("▶️ Execute Image Benchmark", use_container_width=True):
            with st.spinner("Running image benchmark evaluation script..."):
                script_path = BASE_DIR / "scripts" / "evaluate_image.py"
                result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Image evaluation completed successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Execution failed:\n{result.stderr}")

    with col_b2:
        st.markdown("#### 📝 Run Text Evaluation")
        if st.button("▶️ Execute Text Benchmark", use_container_width=True):
            with st.spinner("Running text benchmark evaluation script..."):
                script_path = BASE_DIR / "scripts" / "evaluate_text.py"
                result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ Text evaluation completed successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Execution failed:\n{result.stderr}")
