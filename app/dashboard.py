"""Interactive Streamlit dashboard for live PRONOSTIA bearing health & failure flagging."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import streamlit as st

from pronostia.features import FS_HZ, load_snapshot_csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "predictions.parquet"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

STATUS_COLORS = {
    "HEALTHY": "#22c55e",
    "WARNING": "#f59e0b",
    "CRITICAL": "#ef4444",
}

st.set_page_config(
    page_title="PRONOSTIA Bearing Health Monitor",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Industrial Bearing Failure Monitor (PRONOSTIA / FEMTO-ST)")
st.caption("Real-time vibration Health Indicator, Remaining Useful Life (RUL), and Early-Warning Flagging")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(PREDICTIONS_PATH)


df_all = load_predictions()
if df_all.empty:
    st.error(
        "No predictions found yet. Run the pipeline first:\n\n"
        "```bash\n"
        "uv run pronostia-download --synthetic\n"
        "uv run pronostia-train\n"
        "```"
    )
    st.stop()

# Sidebar controls
st.sidebar.header("Asset & Replay Controls")
splits = sorted(df_all["split"].unique())
default_split = "Full_Test_Set" if "Full_Test_Set" in splits else splits[0]
selected_split = st.sidebar.selectbox("Dataset Split", splits, index=splits.index(default_split))

bearings = sorted(df_all.loc[df_all["split"] == selected_split, "bearing"].unique())
selected_bearing = st.sidebar.selectbox("Bearing Asset", bearings)

bdf = (
    df_all[(df_all["split"] == selected_split) & (df_all["bearing"] == selected_bearing)]
    .sort_values("snapshot_idx")
    .reset_index(drop=True)
)

max_idx = len(bdf) - 1
if "play_idx" not in st.session_state or st.session_state.get("last_bearing") != f"{selected_split}/{selected_bearing}":
    st.session_state["play_idx"] = max_idx
    st.session_state["last_bearing"] = f"{selected_split}/{selected_bearing}"

current_idx = st.sidebar.slider(
    "Current Snapshot (10s intervals)",
    min_value=0,
    max_value=max_idx,
    value=int(st.session_state["play_idx"]),
)
st.session_state["play_idx"] = current_idx

simulate = st.sidebar.button("▶ Simulate Live Stream from Start")
if simulate:
    placeholder = st.empty()
    step = max(1, max_idx // 40)
    for i in range(0, max_idx + 1, step):
        row = bdf.iloc[i]
        color = STATUS_COLORS[row["alert_state"]]
        placeholder.markdown(
            f"**Streaming snapshot {i}/{max_idx}** | Elapsed: `{row['elapsed_min']:.1f} min` | "
            f"Health Index: `{row['health_index']*100:.1f}%` | Pred RUL: `{row['pred_rul_min']:.1f} min` | "
            f"Status: <span style='color:{color};font-weight:700'>{row['alert_state']}</span>",
            unsafe_allow_html=True,
        )
        time.sleep(0.08)
    st.session_state["play_idx"] = max_idx
    st.rerun()

# Slice history up to current_idx
hist = bdf.iloc[: current_idx + 1]
cur = bdf.iloc[current_idx]
status = str(cur["alert_state"])
status_color = STATUS_COLORS[status]

# Top KPI Row
c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(
    f"<div style='padding:12px;border-radius:8px;background:{status_color}22;border:2px solid {status_color};text-align:center'>"
    f"<div style='font-size:12px;opacity:0.8'>ALERT STATUS</div>"
    f"<div style='font-size:24px;font-weight:800;color:{status_color}'>{status}</div></div>",
    unsafe_allow_html=True,
)
c2.metric("Health Index (Degradation)", f"{cur['health_index'] * 100:.1f}%", delta=f"{(cur['health_index'] - hist['health_index'].iloc[max(0, current_idx-5)])*100:+.1f}%")
c3.metric("Predicted RUL", f"{cur['pred_rul_min']:.1f} min", delta=f"{cur['pred_rul_pct']*100:.1f}% life left", delta_color="off")
c4.metric("Horiz. Vibration RMS", f"{cur['h_rms']:.2f} g", delta=f"Kurtosis {cur['h_kurtosis']:.1f}", delta_color="off")
c5.metric("Operating Condition", f"{int(cur['rpm'])} RPM", delta=f"{int(cur['load_n'])} N load", delta_color="off")

# Main Charts: Health Index + RUL Countdown
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    subplot_titles=("Health Indicator Trajectory & Failure Thresholds", "Predicted vs. Actual Remaining Useful Life (%)"),
)

fig.add_trace(
    go.Scatter(x=hist["elapsed_min"], y=hist["health_index"] * 100, name="Health Index (%)", line=dict(color="#3b82f6", width=2.5)),
    row=1,
    col=1,
)
fig.add_hline(y=35, line_dash="dash", line_color="#f59e0b", annotation_text="WARNING (35%)", row=1, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="CRITICAL (70%)", row=1, col=1)

# Mark first WARNING and CRITICAL triggers
for state_label, marker_color in (("WARNING", "#f59e0b"), ("CRITICAL", "#ef4444")):
    trig = hist[hist["alert_state"] == state_label]
    if not trig.empty:
        first_t = trig.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=[first_t["elapsed_min"]],
                y=[first_t["health_index"] * 100],
                mode="markers",
                marker=dict(size=12, color=marker_color, symbol="diamond"),
                name=f"First {state_label} Flag",
            ),
            row=1,
            col=1,
        )

fig.add_trace(
    go.Scatter(x=hist["elapsed_min"], y=hist["pred_rul_pct"] * 100, name="Predicted RUL (%)", line=dict(color="#10b981", width=2.5)),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(x=hist["elapsed_min"], y=hist["rul_pct"] * 100, name="True RUL (%)", line=dict(color="#94a3b8", width=1.8, dash="dot")),
    row=2,
    col=1,
)

fig.update_yaxes(title_text="Degradation (%)", range=[0, 105], row=1, col=1)
fig.update_yaxes(title_text="RUL (%)", range=[0, 105], row=2, col=1)
fig.update_xaxes(title_text="Elapsed Time (minutes)", row=2, col=1)
fig.update_layout(height=540, margin=dict(t=40, b=20, l=40, r=20), legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True)

# Raw Vibration Waveform & FFT Spectrum for the currently selected snapshot
st.subheader(f"Live Sensor Diagnostics — Snapshot `{cur['file_name']}`")
raw_csv = RAW_DIR / selected_split / selected_bearing / str(cur["file_name"])
if raw_csv.exists():
    horiz, vert = load_snapshot_csv(raw_csv)
    t_ms = np.arange(len(horiz)) / FS_HZ * 1000.0
    freqs, psd = signal.welch(horiz - np.mean(horiz), fs=FS_HZ, nperseg=min(512, len(horiz)))

    col_w, col_f = st.columns(2)
    with col_w:
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(x=t_ms, y=horiz, name="Horizontal (g)", line=dict(width=1, color="#3b82f6")))
        fig_w.add_trace(go.Scatter(x=t_ms, y=vert, name="Vertical (g)", line=dict(width=1, color="#8b5cf6"), opacity=0.7))
        fig_w.update_layout(title="Raw Vibration Waveform (0.1s @ 25.6 kHz)", xaxis_title="Time (ms)", yaxis_title="Acceleration (g)", height=320)
        st.plotly_chart(fig_w, use_container_width=True)

    with col_f:
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=freqs / 1000.0, y=10 * np.log10(psd + 1e-12), name="Welch PSD", line=dict(color="#ec4899")))
        fig_f.update_layout(title="Power Spectral Density (FFT)", xaxis_title="Frequency (kHz)", yaxis_title="Power (dB)", height=320)
        st.plotly_chart(fig_f, use_container_width=True)
else:
    st.info(f"Raw snapshot file not found at `{raw_csv}`.")

# Model evaluation summary
if METRICS_PATH.exists():
    with st.expander("Test Set Evaluation Metrics"):
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.json(metrics)
