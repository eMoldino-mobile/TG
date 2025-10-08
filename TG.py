import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Tooling Cycle Time Viewer", layout="wide")

st.title("🛠️ Tooling Cycle Time Viewer")
st.markdown("Upload a tooling run data file to explore cycle time, tolerances, and run segmentation.")

uploaded_file = st.file_uploader("Upload Excel file (e.g., MO-442_2025-10-07.xlsx)", type=["xlsx", "xls"])

# Sidebar controls
st.sidebar.header("Graph Controls")
view_by = st.sidebar.selectbox("View by", ["Shot", "Hour", "Day", "Week", "Month", "Year", "Run"], index=2)
reference_type = st.sidebar.selectbox("Reference", ["Approved CT", "Mode CT"], index=0)
run_threshold = st.sidebar.slider("Run threshold (hours)", 1, 24, 8)
L1 = st.sidebar.slider("L1 Tolerance (%)", 1, 10, 5)
L2 = st.sidebar.slider("L2 Tolerance (%)", 5, 25, 10)
show_temp = st.sidebar.checkbox("Show temperature overlay", False)

# Helper functions
def compute_mode(series, rounding=0.1):
    s = (series / rounding).round() * rounding
    return s.mode().iloc[0] if not s.mode().empty else np.nan

def assign_runs(df, threshold_hours=8):
    df = df.sort_values("shot_time")
    gaps = df["shot_time"].diff()
    run_id = (gaps > pd.Timedelta(hours=threshold_hours)).cumsum()
    df["run_id"] = run_id
    return df

def classify_deviation(ct, ref, l1, l2):
    if pd.isna(ct) or pd.isna(ref):
        return "within"
    if ct > ref * (1 + l2 / 100):
        return "hi_l2"
    elif ct > ref * (1 + l1 / 100):
        return "hi_l1"
    elif ct < ref * (1 - l2 / 100):
        return "lo_l2"
    elif ct < ref * (1 - l1 / 100):
        return "lo_l1"
    else:
        return "within"

color_map = {
    "within": "#3498db",
    "hi_l1": "#f3c06a",
    "hi_l2": "#e74c3c",
    "lo_l1": "#e6953a",
    "lo_l2": "#b03a2e",
}

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Detect timestamp and CT columns
    ts_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    ct_col = next((c for c in df.columns if "cycle" in c.lower() or "ct" in c.lower()), None)
    appr_col = next((c for c in df.columns if "approved" in c.lower()), None)

    if not ts_col or not ct_col:
        st.error("Couldn't detect timestamp or cycle time columns.")
    else:
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        df = df.dropna(subset=[ts_col])
        df = df.rename(columns={ts_col: "shot_time", ct_col: "ct_diff_sec"})

        # Assign runs and compute mode
        df = assign_runs(df, run_threshold)
        approved_ct = df[appr_col].dropna().iloc[0] if appr_col and df[appr_col].notna().any() else compute_mode(df["ct_diff_sec"])

        # Recalculate per run if run mode selected
        run_modes = df.groupby('run_id')['ct_diff_sec'].apply(compute_mode)
        df = df.merge(run_modes.rename('mode_ct'), on='run_id', how='left')

        # Select reference dynamically
        if reference_type == "Approved CT":
            df['reference_ct'] = approved_ct
        else:
            if view_by == 'Run':
                df['reference_ct'] = df['mode_ct']
            else:
                df['reference_ct'] = compute_mode(df['ct_diff_sec'])

        # Classify deviations
        df['deviation_class'] = [classify_deviation(ct, ref, L1, L2) for ct, ref in zip(df['ct_diff_sec'], df['reference_ct'])]
        df['color'] = df['deviation_class'].map(color_map)

        # Build Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['shot_time'], y=df['ct_diff_sec'], marker_color=df['color'], name='Cycle Time'))

        # Add reference and tolerance bands
        ref_val = df['reference_ct'].iloc[0]
        upper_l1 = ref_val * (1 + L1 / 100)
        upper_l2 = ref_val * (1 + L2 / 100)
        lower_l1 = ref_val * (1 - L1 / 100)
        lower_l2 = ref_val * (1 - L2 / 100)

        fig.add_hline(y=ref_val, line_color='green', annotation_text=f'{reference_type}', annotation_position='top left')
        fig.add_hrect(y0=lower_l1, y1=upper_l1, fillcolor='lightblue', opacity=0.2, line_width=0, annotation_text='Within')
        fig.add_hrect(y0=lower_l2, y1=lower_l1, fillcolor='orange', opacity=0.2, line_width=0, annotation_text='L1 Lower')
        fig.add_hrect(y0=upper_l1, y1=upper_l2, fillcolor='orange', opacity=0.2, line_width=0, annotation_text='L1 Upper')

        fig.update_layout(
            title=f'Cycle Time Analysis — {reference_type} reference',
            xaxis_title='Time',
            yaxis_title='Cycle Time (sec)',
            showlegend=False,
            height=700,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Detected columns:** Timestamp = `{ts_col}`, Cycle Time = `{ct_col}`, Approved CT = `{appr_col}`")
else:
    st.info("Please upload a file to begin.")
