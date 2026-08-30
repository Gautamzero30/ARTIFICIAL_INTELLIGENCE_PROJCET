"""
Authentica AI — Enterprise Model Performance & Scientific Analytics Dashboard.
"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Performance Analytics — Authentica AI",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"

st.markdown(
    """
    <style>
    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #60A5FA;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Model Performance & Scientific Analytics")
st.markdown("Audited evaluation metrics dynamically generated across held-out benchmark datasets.")


def load_metric_report(modality: str):
    metric_file = METRICS_DIR / f"{modality}_metrics.json"
    if not metric_file.exists():
        return None
    try:
        with open(metric_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def render_plotly_confusion_matrix(cm_data, title="Confusion Matrix"):
    cm = np.array(cm_data)
    labels = ["Human (0)", "AI (1)"]
    z = cm
    z_text = [[f"<b>{val}</b>" for val in row] for row in cm]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=z_text,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 14, "color": "#F8FAFC"}},
        xaxis_title="Predicted Class",
        yaxis_title="True Ground Truth",
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans", "color": "#CBD5E1"},
    )
    return fig


def render_plotly_roc_curve(curves_data, auc_score, title="ROC Curve"):
    if not curves_data or "roc_curve" not in curves_data:
        return None

    fpr = curves_data["roc_curve"]["fpr"]
    tpr = curves_data["roc_curve"]["tpr"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"Detector (AUC = {auc_score:.3f})" if auc_score else "ROC Curve",
            line=dict(color="#3B82F6", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess (AUC = 0.50)",
            line=dict(color="#64748B", dash="dash"),
        )
    )
    fig.update_layout(
        title={"text": f"<b>{title}</b>", "font": {"size": 14, "color": "#F8FAFC"}},
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Plus Jakarta Sans", "color": "#CBD5E1"},
        legend=dict(x=0.45, y=0.15, bgcolor="rgba(15, 23, 42, 0.7)"),
    )
    return fig


# Top Executive Overview Table
st.markdown("### 🏆 Executive Multi-Modality Benchmark Summary")

img_rep = load_metric_report("image")
txt_rep = load_metric_report("text")
aud_rep = load_metric_report("audio")

summary_rows = []
for name, rep in [("🖼️ Image (ViT)", img_rep), ("📝 Text (RoBERTa)", txt_rep), ("🎵 Audio (Wav2Vec2)", aud_rep)]:
    if rep:
        m = rep["metrics"]
        summary_rows.append({
            "Modality": name,
            "Model Backbone": rep.get("model_name"),
            "Samples": m["sample_count"],
            "Accuracy": f"{m['accuracy']*100:.1f}%",
            "Precision": f"{m['precision']*100:.1f}%",
            "Recall": f"{m['recall']*100:.1f}%",
            "F1-Score": f"{m['f1_score']*100:.1f}%",
            "ROC-AUC": f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "N/A",
            "Status": "✅ Evaluated",
        })
    else:
        summary_rows.append({
            "Modality": name,
            "Model Backbone": "Pending",
            "Samples": 0,
            "Accuracy": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-Score": "N/A",
            "ROC-AUC": "N/A",
            "Status": "⚠️ Pending Run",
        })

df_summary = pd.DataFrame(summary_rows)
st.dataframe(df_summary, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# Deep-dive Modality Tabs
tab_img, tab_txt, tab_aud, tab_rob, tab_runner = st.tabs([
    "🖼️ Image Forensics",
    "📝 Text Intelligence",
    "🎵 Audio & Voice",
    "🛡️ Robustness Testing",
    "⚙️ Live Benchmark Runner",
])

with tab_img:
    if img_rep:
        m = img_rep["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="kpi-card"><div class="kpi-title">Accuracy</div><div class="kpi-value">{m["accuracy"]*100:.1f}%</div><div class="kpi-sub">Held-out test set</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div class="kpi-title">Precision</div><div class="kpi-value">{m["precision"]*100:.1f}%</div><div class="kpi-sub">Low false alarm rate</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div class="kpi-title">Recall</div><div class="kpi-value">{m["recall"]*100:.1f}%</div><div class="kpi-sub">Synthetic catch rate</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi-card"><div class="kpi-title">ROC-AUC</div><div class="kpi-value">{m["roc_auc"]:.3f}</div><div class="kpi-sub">Discriminative power</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_cm, c_roc = st.columns(2)
        with c_cm:
            st.plotly_chart(render_plotly_confusion_matrix(m["confusion_matrix"], "Image Confusion Matrix"), use_container_width=True)
        with c_roc:
            roc_fig = render_plotly_roc_curve(img_rep.get("curves"), m.get("roc_auc"), "Image ROC Curve")
            if roc_fig:
                st.plotly_chart(roc_fig, use_container_width=True)
    else:
        st.info("Run Image benchmark to view metrics.")

with tab_txt:
    if txt_rep:
        m = txt_rep["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="kpi-card"><div class="kpi-title">Accuracy</div><div class="kpi-value">{m["accuracy"]*100:.1f}%</div><div class="kpi-sub">HC3 benchmark</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div class="kpi-title">Precision</div><div class="kpi-value">{m["precision"]*100:.1f}%</div><div class="kpi-sub">Positive predictive value</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div class="kpi-title">Recall</div><div class="kpi-value">{m["recall"]*100:.1f}%</div><div class="kpi-sub">ChatGPT detection rate</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi-card"><div class="kpi-title">ROC-AUC</div><div class="kpi-value">{m["roc_auc"]:.3f}</div><div class="kpi-sub">Discriminative power</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_cm, c_roc = st.columns(2)
        with c_cm:
            st.plotly_chart(render_plotly_confusion_matrix(m["confusion_matrix"], "Text Confusion Matrix"), use_container_width=True)
        with c_roc:
            roc_fig = render_plotly_roc_curve(txt_rep.get("curves"), m.get("roc_auc"), "Text ROC Curve")
            if roc_fig:
                st.plotly_chart(roc_fig, use_container_width=True)
    else:
        st.info("Run Text benchmark to view metrics.")

with tab_aud:
    if aud_rep:
        m = aud_rep["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="kpi-card"><div class="kpi-title">Accuracy</div><div class="kpi-value">{m["accuracy"]*100:.1f}%</div><div class="kpi-sub">Speech benchmark</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div class="kpi-title">Precision</div><div class="kpi-value">{m["precision"]*100:.1f}%</div><div class="kpi-sub">Voice clone precision</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div class="kpi-title">Recall</div><div class="kpi-value">{m["recall"]*100:.1f}%</div><div class="kpi-sub">Deepfake catch rate</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi-card"><div class="kpi-title">ROC-AUC</div><div class="kpi-value">{m["roc_auc"]:.3f}</div><div class="kpi-sub">Discriminative power</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c_cm, c_roc = st.columns(2)
        with c_cm:
            st.plotly_chart(render_plotly_confusion_matrix(m["confusion_matrix"], "Audio Confusion Matrix"), use_container_width=True)
        with c_roc:
            roc_fig = render_plotly_roc_curve(aud_rep.get("curves"), m.get("roc_auc"), "Audio ROC Curve")
            if roc_fig:
                st.plotly_chart(roc_fig, use_container_width=True)
    else:
        st.info("Run Audio benchmark to view metrics.")

with tab_rob:
    st.markdown("#### 🛡️ Robustness Under Real-World Perturbations")
    rob_file = EXPERIMENTS_DIR / "robustness_results.json"
    if rob_file.exists():
        with open(rob_file, "r", encoding="utf-8") as f:
            rob_data = json.load(f)
        
        img_rob = rob_data.get("image_robustness", {})
        txt_rob = rob_data.get("text_robustness", {})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🖼️ Image: JPEG Compression (Q=30)")
            st.write(f"• Original Accuracy: **{img_rob.get('original_accuracy', 0)*100:.1f}%**")
            st.write(f"• Perturbed Accuracy: **{img_rob.get('compressed_accuracy', 0)*100:.1f}%**")
            st.write(f"• Original ROC-AUC: **{img_rob.get('original_roc_auc', 0):.3f}**")

        with c2:
            st.markdown("##### 📝 Text: Aggressive Truncation (~65 chars)")
            st.write(f"• Original Accuracy: **{txt_rob.get('original_accuracy', 0)*100:.1f}%**")
            st.write(f"• Perturbed Accuracy: **{txt_rob.get('truncated_accuracy', 0)*100:.1f}%**")
            st.write(f"• Original ROC-AUC: **{txt_rob.get('original_roc_auc', 0):.3f}**")
    else:
        st.info("Execute `python scripts/evaluate_robustness.py` to view perturbation findings.")

with tab_runner:
    st.markdown("#### ⚙️ Re-Execute Benchmarks Live")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("▶️ Run Image Benchmark", use_container_width=True):
            subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "evaluate_image.py")])
            st.success("Image benchmark updated!")
            st.rerun()
    with b2:
        if st.button("▶️ Run Text Benchmark", use_container_width=True):
            subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "evaluate_text.py")])
            st.success("Text benchmark updated!")
            st.rerun()
    with b3:
        if st.button("▶️ Run Audio Benchmark", use_container_width=True):
            subprocess.run([sys.executable, str(BASE_DIR / "scripts" / "evaluate_audio.py")])
            st.success("Audio benchmark updated!")
            st.rerun()
