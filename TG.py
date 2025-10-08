import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Tooling Cycle Time Viewer", layout="wide")

st.title("🛠️ Tooling Cycle Time Viewer")
st.caption("Upload a tooling Excel file. Bars use Actual CT with Run Rate substitution (Δt > prev Actual CT + 2s → use Δt).")

uploaded_file = st.file_uploader(
    "Upload Excel file", type=["xlsx", "xls"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Graph Controls")
view_by = st.sidebar.selectbox(
    "View by",
    ["Shot", "Hour", "Day", "Week", "Month", "Year", "Run"],
    index=2,
)
reference_type = st.sidebar.selectbox("Reference", ["Approved CT", "Mode CT"], index=0)
run_threshold = st.sidebar.slider("Run threshold (hours)", 1, 24, 8)
L1 = st.sidebar.slider("L1 Tolerance (%)", 1, 10, 5)
L2 = st.sidebar.slider("L2 Tolerance (%)", 5, 25, 10)
hide_extreme = st.sidebar.checkbox("Hide extreme stops (clip outliers)", value=True)
use_log_y = st.sidebar.checkbox("Use logarithmic Y scale", value=False)

# colors
COLOR = {
    "within": "#3498db",
    "hi_l1": "#f3c06a",
    "hi_l2": "#e74c3c",
    "lo_l1": "#e6953a",
    "lo_l2": "#b03a2e",
    "approved": "#27ae60",
    "mode": "#0b3c83",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def compute_mode(series: pd.Series, rounding: float = 0.1) -> float:
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
    ts = next((c for c in df.columns if "shot time" in c.lower()), None)
    if ts is None:
        ts = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    actual_ct = next((c for c in df.columns if "actual" in c.lower() and "ct" in c.lower()), None)
    approved_ct = next((c for c in df.columns if "approved" in c.lower() and "ct" in c.lower()), None)
    temp_col = next((c for c in df.columns if "temp" in c.lower()), None)
    return ts, actual_ct, approved_ct, temp_col

def resample_view(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    g = df.set_index("shot_time").sort_index()
    out = pd.DataFrame(index=g.resample(freq).mean().index)
    out["mean_ct"] = g["ct_use"].resample(freq).mean()
    out["shots"]   = g["ct_use"].resample(freq).count()
    out["mode_ct"] = g["ct_use"].resample(freq).apply(lambda s: compute_mode(s, 0.1))
    out = out.dropna(how="all").reset_index().rename(columns={"shot_time": "bucket"})
    return out

def build_bands(fig, x0, x1, ref, l1, l2, label_prefix="Bands"):
    u1 = ref * (1 + l1 / 100.0)
    u2 = ref * (1 + l2 / 100.0)
    l1v = ref * (1 - l1 / 100.0)
    l2v = ref * (1 - l2 / 100.0)
    # Within band
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=l1v, y1=u1,
                  fillcolor="lightblue", opacity=0.2, line_width=0, name=f"{label_prefix} Within")
    # L1 bands (upper/lower)
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=u1, y1=u2,
                  fillcolor="#f3c06a", opacity=0.22, line_width=0, name=f"{label_prefix} L1 Upper")
    fig.add_shape(type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=l2v, y1=l1v,
                  fillcolor="#e6953a", opacity=0.22, line_width=0, name=f"{label_prefix} L1 Lower")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file:
    raw = pd.read_excel(uploaded_file)
    ts_col, actual_ct_col, approved_col, temp_col = detect_columns(raw)

    if not ts_col or not actual_ct_col:
        st.error("Couldn't detect required columns. Need a timestamp column and an 'Actual CT' column.")
        st.stop()

    df = raw.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    df = df.rename(columns={ts_col: "shot_time", actual_ct_col: "actual_ct"})
    df = df.sort_values("shot_time").reset_index(drop=True)

    # Run Rate substitution: ct_use = Δt if Δt > prev_actual + 2s else prev_actual
    time_diff_sec = df["shot_time"].diff().dt.total_seconds()
    prev_actual = df["actual_ct"].shift(1)
    rounding_buffer = 2.0
    use_gap = time_diff_sec > (prev_actual + rounding_buffer)
    df["ct_use"] = np.where(use_gap, time_diff_sec, prev_actual)
    if pd.isna(df.loc[0, "ct_use"]):
        df.loc[0, "ct_use"] = df.loc[0, "actual_ct"]

    # Runs (uses shot_time gaps)
    df = assign_runs(df, run_threshold)

    # Approved reference (fallback to mode if missing)
    if approved_col and approved_col in df.columns and df[approved_col].notna().any():
        approved_ct_val = float(df[approved_col].dropna().iloc[0])
    else:
        approved_ct_val = compute_mode(df["ct_use"])

    # Outlier clipping (apply BEFORE view aggregation)
    clip_threshold = approved_ct_val * 3.0
    if hide_extreme:
        df_plot = df[df["ct_use"] <= clip_threshold].copy()
    else:
        df_plot = df.copy()

    # Choose reference for tolerance bands
    if reference_type == "Approved CT":
        reference_val = approved_ct_val
    else:
        if view_by == "Run":
            # per-run mode
            reference_val = np.nan  # per-run, handled in run plot
        else:
            reference_val = compute_mode(df_plot["ct_use"])

    # Build figure per view
    if view_by == "Shot":
        fig = go.Figure()
        # classify by current reference_val
        classes = [classify(ct, reference_val, L1, L2) for ct in df_plot["ct_use"]]
        colors = [COLOR[c] for c in classes]
        fig.add_bar(x=df_plot["shot_time"], y=df_plot["ct_use"], marker_color=colors, name="Cycle Time")

        # bands + ref line
        x0, x1 = df_plot["shot_time"].min(), df_plot["shot_time"].max()
        build_bands(fig, x0, x1, reference_val, L1, L2)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[reference_val, reference_val],
                                 mode="lines", line=dict(color=COLOR["approved"] if reference_type=="Approved CT" else COLOR["mode"]),
                                 name=f"{reference_type}"))
        # right axis shot count (cumulative by minute)
        counts = df_plot.set_index("shot_time")["ct_use"].resample("15min").count()
        fig.add_trace(go.Scatter(x=counts.index, y=counts.values, name="Shots (15m)", line=dict(dash="dot")), secondary_y=True)

    elif view_by in ["Hour", "Day", "Week", "Month", "Year"]:
        freq_map = {"Hour": "H", "Day": "D", "Week": "W", "Month": "M", "Year": "Y"}
        freq = freq_map[view_by]
        view_df = resample_view(df_plot, freq)
        fig = go.Figure()
        # CT line
        fig.add_trace(go.Scatter(x=view_df["bucket"], y=view_df["mean_ct"],
                                 mode="lines+markers", name=f"Mean CT ({view_by})",
                                 line=dict(color="#2c3e50")))
        # Shots bar (secondary y)
        fig.add_trace(go.Bar(x=view_df["bucket"], y=view_df["shots"], name=f"Shots ({view_by})",
                             opacity=0.35), secondary_y=True)
        # bands across full x-range using reference_val
        if not view_df.empty:
            x0, x1 = view_df["bucket"].min(), view_df["bucket"].max()
            build_bands(fig, x0, x1, reference_val, L1, L2)
            fig.add_trace(go.Scatter(x=[x0, x1], y=[reference_val, reference_val],
                                     mode="lines", line=dict(color=COLOR["approved"] if reference_type=="Approved CT" else COLOR["mode"]),
                                     name=f"{reference_type}"))

    else:  # Run view
        runs = (
            df_plot.groupby("run_id")
                  .agg(start=("shot_time", "min"),
                       end=("shot_time", "max"),
                       mean_ct=("ct_use", "mean"),
                       shots=("ct_use", "count"),
                       mode_ct=("ct_use", lambda s: compute_mode(s, 0.1)))
                  .reset_index()
        )
        fig = go.Figure()
        # Mean CT per run
        fig.add_trace(go.Scatter(x=runs["start"], y=runs["mean_ct"], mode="lines+markers",
                                 name="Mean CT (Run)"))
        # Shots per run (secondary y)
        fig.add_trace(go.Bar(x=runs["start"], y=runs["shots"], name="Shots (Run)", opacity=0.35),
                      secondary_y=True)
        # Bands
        if reference_type == "Approved CT":
            if not runs.empty:
                x0, x1 = runs["start"].min(), runs["end"].max()
                build_bands(fig, x0, x1, approved_ct_val, L1, L2)
                fig.add_trace(go.Scatter(x=[x0, x1], y=[approved_ct_val, approved_ct_val],
                                         mode="lines", line=dict(color=COLOR["approved"]), name="Approved CT"))
        else:
            # per-run mode bands
            for _, r in runs.iterrows():
                build_bands(fig, r["start"], r["end"], r["mode_ct"], L1, L2, label_prefix=f"Run {int(r['run_id'])} Bands")
            fig.add_trace(go.Scatter(x=runs["start"], y=runs["mode_ct"], mode="lines",
                                     line=dict(color=COLOR["mode"], dash="dot"), name="Mode CT (Run)"))

    # Layout and axes
    fig.update_layout(
        title=f"Cycle Time Analysis — {reference_type} reference — View: {view_by}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=90),
    )
    fig.update_yaxes(title_text="Cycle Time (sec)", type="log" if use_log_y else "linear", secondary_y=False)
    fig.update_yaxes(title_text="Shot Count", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # Data preview
    with st.expander("Data preview (first 12 rows)"):
        st.dataframe(df.head(12))

    st.caption(
        f"Detected columns — Timestamp: `{ts_col}`, Actual CT: `{actual_ct_col}`, Approved CT: `{approved_col}`. "
        f"{'Outliers hidden (> 3× Approved CT).' if hide_extreme else 'Outliers visible.'}"
    )

else:
    st.info("Please upload a file to begin.")