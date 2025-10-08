import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

st.set_page_config(page_title="Tooling Cycle Time Viewer", layout="wide")

st.title("🛠️ Tooling Cycle Time Viewer")
st.caption("Upload a tooling Excel file to analyze cycle times, stops, and production runs.")

uploaded_file = st.file_uploader(
    "Upload Excel file", type=["xlsx", "xls"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Graph Controls")
# This is a placeholder; it will be updated once a file is loaded.
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(date.today(), date.today()),
)

view_by = st.sidebar.selectbox(
    "Data View",
    ["Every Shot", "Mode per Hour", "Mode per Day", "By Run"],
    index=1,  # Default to "Mode per Hour"
)
reference_type = st.sidebar.selectbox("Reference", ["Approved CT", "Mode CT"], index=0)
run_threshold = st.sidebar.slider("Run threshold (hours)", 1, 24, 8)
L1 = st.sidebar.slider("L1 Tolerance (%)", 1, 10, 5)
L2 = st.sidebar.slider("L2 Tolerance (%)", 5, 25, 10)
hide_extreme = st.sidebar.checkbox("Hide extreme stops (clip outliers)", value=True)
use_log_y = st.sidebar.checkbox("Use logarithmic Y scale", value=False)

# colors
COLOR = {
    "within": "#3498db",    # Blue
    "hi_l1": "#F5B7B1",     # Pastel Red
    "hi_l2": "#E74C3C",     # Dark Red
    "lo_l1": "#FAD7A0",     # Pastel Orange
    "lo_l2": "#E6953A",     # Orange
    "approved": "#27ae60",  # Green
    "mode": "#0b3c83",      # Dark Blue
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def compute_mode(series: pd.Series, rounding: float = 0.1) -> float:
    if series.empty:
        return np.nan
    s = (series / rounding).round() * rounding
    m = s.mode()
    return float(m.iloc[0]) if not m.empty else np.nan

def assign_runs(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    df = df.sort_values("shot_time").copy()
    gaps = df["shot_time"].diff()
    df["run_id"] = (gaps > pd.Timedelta(hours=hours)).cumsum()
    return df

def classify(ct: float, ref: float, l1: float, l2: float) -> str:
    if pd.isna(ct) or pd.isna(ref):
        return "within"
    if ct > ref * (1 + l2 / 100):
        return "hi_l2"
    if ct > ref * (1 + l1 / 100):
        return "hi_l1"
    if ct < ref * (1 - l2 / 100):
        return "lo_l2"
    if ct < ref * (1 - l1 / 100):
        return "lo_l1"
    return "within"

def detect_columns(df: pd.DataFrame):
    ts = next((c for c in df.columns if "shot time" in c.lower() or "date/time" in c.lower()), None)
    if ts is None:
        ts = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    actual_ct = next((c for c in df.columns if "actual" in c.lower() and "ct" in c.lower()), None)
    approved_ct = next((c for c in df.columns if "approved" in c.lower() and "ct" in c.lower()), None)
    temp_col = next((c for c in df.columns if "temp" in c.lower()), None)
    return ts, actual_ct, approved_ct, temp_col

def resample_view(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Safely resample cycle time data to the given frequency."""
    if df.empty or "shot_time" not in df.columns:
        return pd.DataFrame(columns=["bucket", "mean_ct", "shots", "mode_ct"])
    g = df.copy()
    g["shot_time"] = pd.to_datetime(g["shot_time"], errors="coerce")
    g = g.dropna(subset=["shot_time"]).sort_values("shot_time").set_index("shot_time")
    if "ct_use" not in g.columns:
        return pd.DataFrame(columns=["bucket", "mean_ct", "shots", "mode_ct"])
    try:
        mean_ct = g["ct_use"].resample(freq).mean()
        shot_count = g["ct_use"].resample(freq).count()
        mode_ct = g["ct_use"].resample(freq).apply(lambda s: compute_mode(s, 0.1))
        out = pd.DataFrame({"bucket": mean_ct.index, "mean_ct": mean_ct.values, "shots": shot_count.values, "mode_ct": mode_ct.values})
        return out.dropna(subset=["mean_ct"])
    except Exception as e:
        st.warning(f"⚠️ Resampling failed ({e.__class__.__name__}).")
        return pd.DataFrame(columns=["bucket", "mean_ct", "shots", "mode_ct"])

def build_bands(fig, x0, x1, ref, l1, l2, y_min, y_max):
    u1 = ref * (1 + l1 / 100.0)
    u2 = ref * (1 + l2 / 100.0)
    l1v = ref * (1 - l1 / 100.0)
    l2v = ref * (1 - l2 / 100.0)
    
    # L1 Upper (Pastel Red)
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=u1, y1=u2, fillcolor=COLOR["hi_l1"], opacity=0.3, line_width=0, layer="below")
    # L2 Upper (Dark Red)
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=u2, y1=y_max, fillcolor=COLOR["hi_l2"], opacity=0.3, line_width=0, layer="below")
    # L1 Lower (Pastel Orange)
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=l2v, y1=l1v, fillcolor=COLOR["lo_l1"], opacity=0.3, line_width=0, layer="below")
    # L2 Lower (Orange)
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y_min, y1=l2v, fillcolor=COLOR["lo_l2"], opacity=0.3, line_width=0, layer="below")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file:
    raw = pd.read_excel(uploaded_file)
    ts_col, actual_ct_col, approved_col, temp_col = detect_columns(raw)

    if not ts_col or not actual_ct_col:
        st.error("Couldn't detect required columns. Need a timestamp column (e.g., 'Date/Time') and an 'Actual CT' column.")
        st.stop()

    df = raw.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.rename(columns={ts_col: "shot_time", actual_ct_col: "actual_ct"})
    df = df.sort_values("shot_time").reset_index(drop=True)

    # Date Range Filter
    min_date = df["shot_time"].min().date()
    max_date = df["shot_time"].max().date()
    date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]).replace(hour=23, minute=59, second=59)
        df = df[(df['shot_time'] >= start_date) & (df['shot_time'] <= end_date)]
    
    if df.empty:
        st.warning("No data available for the selected date range.")
        st.stop()

    # **NEW STOP CYCLE LOGIC**: Assign stop time to the cycle *before* the stop.
    df['ct_use'] = df['actual_ct'].copy()
    time_diff_sec = df["shot_time"].diff().dt.total_seconds()
    rounding_buffer = 2.0
    is_a_stop = time_diff_sec > (df["actual_ct"].shift(1) + rounding_buffer)
    stop_start_indices = df.index[is_a_stop] - 1
    valid_stop_indices = stop_start_indices[stop_start_indices >= 0]
    df.loc[valid_stop_indices, 'ct_use'] = time_diff_sec[is_a_stop].values

    df = assign_runs(df, run_threshold)

    if approved_col and approved_col in df.columns and df[approved_col].notna().any():
        approved_ct_val = float(df[approved_col].dropna().iloc[0])
    else:
        approved_ct_val = compute_mode(df["ct_use"])

    clip_threshold = approved_ct_val * 5.0 
    if hide_extreme:
        df_plot = df[df["ct_use"] <= clip_threshold].copy()
    else:
        df_plot = df.copy()

    if reference_type == "Approved CT":
        reference_val = approved_ct_val
    else:
        if view_by == "By Run":
            reference_val = np.nan
        else:
            reference_val = compute_mode(df_plot["ct_use"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    y_range = [0, df_plot["ct_use"].max() * 1.1] # For band boundaries

    # Build figure based on view
    view_map = {"Every Shot": "Shot", "Mode per Hour": "Hour", "Mode per Day": "Day", "By Run": "Run"}
    current_view = view_map[view_by]
    
    if current_view == "Shot":
        classes = [classify(ct, reference_val, L1, L2) for ct in df_plot["ct_use"]]
        colors = [COLOR[c] for c in classes]
        fig.add_bar(x=df_plot["shot_time"], y=df_plot["ct_use"], marker_color=colors, name="Cycle Time")

        if not df_plot.empty:
            x0, x1 = df_plot["shot_time"].min(), df_plot["shot_time"].max()
            build_bands(fig, x0, x1, reference_val, L1, L2, y_range[0], y_range[1])
            fig.add_trace(go.Scatter(x=[x0, x1], y=[reference_val, reference_val], mode="lines", line=dict(color=COLOR["approved"] if reference_type=="Approved CT" else COLOR["mode"], dash='dash'), name=f"{reference_type}"))
        
        df_plot['shot_num_cumulative'] = range(1, len(df_plot) + 1)
        fig.add_trace(go.Scatter(x=df_plot["shot_time"], y=df_plot['shot_num_cumulative'], name="Cumulative Shots", line=dict(dash="dot")), secondary_y=True)

    elif current_view in ["Hour", "Day"]:
        freq_map = {"Hour": "H", "Day": "D"}
        view_df = resample_view(df_plot, freq_map[current_view])
        
        if not view_df.empty:
            fig.add_trace(go.Scatter(x=view_df["bucket"], y=view_df["mode_ct"], mode="lines+markers", name=f"Mode CT ({current_view})", line=dict(color="#2c3e50")))
            fig.add_trace(go.Bar(x=view_df["bucket"], y=view_df["shots"], name=f"Shots ({current_view})", opacity=0.35), secondary_y=True)
            x0, x1 = view_df["bucket"].min(), view_df["bucket"].max()
            build_bands(fig, x0, x1, reference_val, L1, L2, y_range[0], y_range[1])
            fig.add_trace(go.Scatter(x=[x0, x1], y=[reference_val, reference_val], mode="lines", line=dict(color=COLOR["approved"] if reference_type=="Approved CT" else COLOR["mode"], dash='dash'), name=f"{reference_type}"))

    elif current_view == "Run":
        runs = (df_plot.groupby("run_id").agg(start=("shot_time", "min"), end=("shot_time", "max"), mean_ct=("ct_use", "mean"), shots=("ct_use", "count"), mode_ct=("ct_use", lambda s: compute_mode(s, 0.1))).reset_index())
        fig.add_trace(go.Scatter(x=runs["start"], y=runs["mean_ct"], mode="lines+markers", name="Mean CT (Run)"))
        fig.add_trace(go.Bar(x=runs["start"], y=runs["shots"], name="Shots (Run)", opacity=0.35), secondary_y=True)

        if reference_type == "Approved CT":
            if not runs.empty:
                x0, x1 = runs["start"].min(), runs["end"].max()
                build_bands(fig, x0, x1, approved_ct_val, L1, L2, y_range[0], y_range[1])
                fig.add_trace(go.Scatter(x=[x0, x1], y=[approved_ct_val, approved_ct_val], mode="lines", line=dict(color=COLOR["approved"], dash='dash'), name="Approved CT"))
        else: # per-run mode bands
            for _, r in runs.iterrows():
                if pd.notna(r["mode_ct"]):
                    build_bands(fig, r["start"], r["end"], r["mode_ct"], L1, L2, y_range[0], y_range[1])
            fig.add_trace(go.Scatter(x=runs["start"], y=runs["mode_ct"], mode="lines", line=dict(color=COLOR["mode"], dash="dot"), name="Mode CT (per Run)"))

    # Layout and axes
    fig.update_layout(
        title=f"Cycle Time Analysis — Reference: {reference_type} — View: {view_by}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=90),
    )
    fig.update_yaxes(title_text="Cycle Time (sec)", type="log" if use_log_y else "linear", secondary_y=False, range=[y_range[0], y_range[1] if not use_log_y else None])
    fig.update_yaxes(title_text="Shot Count", secondary_y=True, showgrid=False)
    
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data preview (first 12 rows of filtered & processed data)"):
        st.dataframe(df.head(12))

    st.caption(f"Detected columns — Timestamp: `{ts_col}`, Actual CT: `{actual_ct_col}`, Approved CT: `{approved_col}`. {'Outliers hidden.' if hide_extreme else 'Outliers visible.'}")

else:
    st.info("Please upload an Excel file to begin.")

