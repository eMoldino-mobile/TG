import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Warranty & EOL Shot Filter")

# --- Color Constants ---
STATUS_COLORS = {
    'Good': '#2ecc71',
    'Scrap': '#e74c3c',
    'Startup - Bad': '#f39c12',
    'Bad (Post-Pause)': '#3498db'
}

# --- Core Calculation Class ---
class WarrantyFilter:
    """
    Analyzes shot data to filter shots for warranty and end-of-life purposes.
    It expects a DataFrame with standardized column names (lowercase, snake_case).
    """
    def __init__(self, df: pd.DataFrame, approved_ct: float, pause_minutes: int,
                 ct_upper_pct: float, ct_lower_pct: float, startup_shots_count: int,
                 stable_period_shots: int):
        self.df_raw = df.copy()
        self.approved_ct = approved_ct
        self.pause_minutes = pause_minutes
        self.ct_upper_limit = approved_ct * (1 + ct_upper_pct / 100)
        self.ct_lower_limit = approved_ct * (1 - ct_lower_pct / 100)
        self.startup_shots_count = startup_shots_count
        self.stable_period_shots = stable_period_shots
        self.results = self._analyze_shots()

    def _prepare_data(self) -> pd.DataFrame:
        """Prepares the raw DataFrame for analysis."""
        df = self.df_raw.copy()
        # Expects standardized columns 'shot_time' and 'actual_ct'
        if "shot_time" not in df.columns or "actual_ct" not in df.columns:
            st.error("Input data must contain 'SHOT TIME' and 'Actual CT' columns.")
            return pd.DataFrame()

        df["shot_time"] = pd.to_datetime(df["shot_time"], errors="coerce")
        df = df.dropna(subset=["shot_time", "actual_ct"]).sort_values("shot_time").reset_index(drop=True)
        
        if df.empty:
            return pd.DataFrame()
            
        df["time_diff_minutes"] = df["shot_time"].diff().dt.total_seconds() / 60
        df["time_diff_minutes"].fillna(0, inplace=True)
        return df

    def _analyze_shots(self) -> dict:
        """Runs the full analysis and classification logic."""
        df = self._prepare_data()
        if df.empty:
            return {}

        # --- Initial Classification ---
        df['part_status'] = 'Good'
        df['affects_warranty'] = True
        
        is_bad_cycle = (df['actual_ct'] > self.ct_upper_limit) | (df['actual_ct'] < self.ct_lower_limit)
        df.loc[is_bad_cycle, 'part_status'] = 'Bad (Post-Pause)'

        # --- Identify and Re-classify Startup Shots ---
        df['is_pause_before'] = df['time_diff_minutes'] > self.pause_minutes
        pause_indices = df[df['is_pause_before']].index

        for idx in pause_indices:
            startup_range_start = idx
            startup_range_end = min(idx + self.startup_shots_count, len(df))
            
            for startup_idx in range(startup_range_start, startup_range_end):
                if is_bad_cycle.at[startup_idx]:
                    df.loc[startup_idx, 'part_status'] = 'Startup - Bad'
                    df.loc[startup_idx, 'affects_warranty'] = False
        
        # --- Identify and Re-classify Scrap Shots ---
        df['pause_in_window'] = df['is_pause_before'].rolling(window=self.stable_period_shots, min_periods=1).max()
        is_stable = df['pause_in_window'] != 1.0

        is_scrap = (df['part_status'] == 'Bad (Post-Pause)') & is_stable
        df.loc[is_scrap, 'part_status'] = 'Scrap'

        # --- Final Metrics Calculation ---
        total_shots = len(df)
        summary = {
            "Total Accumulated Shots": total_shots,
            "Adjusted Accumulated Shots (Affects Warranty)": int(df['affects_warranty'].sum()),
            "Good Parts": (df['part_status'] == 'Good').sum(),
            "Scrap Parts": (df['part_status'] == 'Scrap').sum(),
            "Discounted Startup Shots (Bad)": (df['part_status'] == 'Startup - Bad').sum(),
            "Other Bad Cycles (Post-Pause)": (df['part_status'] == 'Bad (Post-Pause)').sum()
        }
        
        return {"processed_df": df, "summary_metrics": summary}

# --- Plotting Function ---
def plot_shot_analysis(df, approved_ct, upper_limit, lower_limit):
    """Creates a Plotly bar chart of the shot analysis."""
    if df.empty:
        st.info("No data to display for the selected date.")
        return go.Figure()

    fig = go.Figure()

    # --- Create the main bar chart trace with correct colors ---
    df['color'] = df['part_status'].map(STATUS_COLORS)
    fig.add_trace(go.Bar(
        x=df['shot_time'],
        y=df['actual_ct'],
        marker_color=df['color'],
        name='Cycle Time',
        showlegend=False
    ))

    # --- Add dummy traces for a custom legend ---
    for status, color in STATUS_COLORS.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            marker_color=color,
            name=status,
            showlegend=True
        ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='purple', dash='dash', width=1.5),
        name='Production Pause',
        showlegend=True
    ))

    # Add tolerance lines
    fig.add_hline(y=approved_ct, line_dash="dot", line_color="grey",
                  annotation_text="Approved CT", annotation_position="bottom right")
    fig.add_hline(y=upper_limit, line_dash="dash", line_color="orange",
                  annotation_text="Upper Tolerance", annotation_position="top right")
    fig.add_hline(y=lower_limit, line_dash="dash", line_color="orange",
                  annotation_text="Lower Tolerance", annotation_position="bottom right")
    
    # Add vertical lines for production pauses
    pause_times = df[df['is_pause_before']]['shot_time']
    for p_time in pause_times:
        fig.add_vline(x=p_time, line_width=1.5, line_dash="dash", line_color="purple")

    # Set y-axis range dynamically
    y_axis_max = max(df['actual_ct'].max() * 1.1, upper_limit * 1.5) if not df.empty else approved_ct * 2
    
    fig.update_layout(
        title="Shot-by-Shot Cycle Time Analysis",
        xaxis_title="Shot Time",
        yaxis_title="Actual Cycle Time (seconds)",
        yaxis_range=[0, y_axis_max],
        legend_title_text='Legend',
        bargap=0.1
    )
    return fig

# --- Main App UI ---
st.title("Warranty & End-of-Life (EOL) Shot Filter")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Analysis Controls")
uploaded_file = st.sidebar.file_uploader("Upload your shot data Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    
    # Standardize column names to handle different cases (e.g., 'SHOT TIME', 'Shot Time', 'shot_time')
    df_input.columns = [col.strip().lower().replace(' ', '_') for col in df_input.columns]

    # --- MANDATORY CHECK for 'approved_ct' column ---
    if 'approved_ct' not in df_input.columns or df_input['approved_ct'].dropna().empty:
        st.error("Error: The uploaded Excel file must contain a column named 'APPROVED CT' with at least one valid number.")
        st.stop()  # Stop the app if the column is missing or empty

    # If the check passes, proceed with loading the file and setting up controls
    st.sidebar.success(f"File '{uploaded_file.name}' loaded successfully!")
    
    # --- Date Filter ---
    df_input['shot_time'] = pd.to_datetime(df_input['shot_time'], errors='coerce')
    df_input.dropna(subset=['shot_time'], inplace=True)
    df_input['date'] = df_input['shot_time'].dt.date
    available_dates = sorted(df_input["date"].unique())
    
    selected_date = st.sidebar.selectbox(
        "Select Date to Analyze",
        options=available_dates,
        index=len(available_dates) - 1,  # Default to the last date
        format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y')
    )

    st.sidebar.markdown("---")
    
    # Use the first valid value from the mandatory 'approved_ct' column
    default_approved_ct = df_input['approved_ct'].dropna().iloc[0]
    st.sidebar.info(f"Using 'Approved CT' from file: {default_approved_ct:.2f}s")

    approved_ct = st.sidebar.number_input(
        "Approved Cycle Time (seconds)",
        min_value=0.1, value=float(default_approved_ct), step=0.1,
        help="This value is automatically populated from the 'APPROVED CT' column in your file."
    )
    
    with st.sidebar.expander("Threshold Settings", expanded=True):
        pause_minutes = st.slider(
            "Production Pause Threshold (minutes)",
            min_value=1, max_value=480, value=30,
            help="Any gap between shots longer than this is considered a production pause."
        )
        ct_upper_pct = st.slider(
            "Cycle Time Upper Tolerance (%)",
            min_value=1, max_value=100, value=15,
            help="Shots with a cycle time this much % above 'Approved CT' are flagged."
        )
        ct_lower_pct = st.slider(
            "Cycle Time Lower Tolerance (%)",
            min_value=1, max_value=100, value=15,
            help="Shots with a cycle time this much % below 'Approved CT' are flagged."
        )

    with st.sidebar.expander("Warranty & Scrap Logic Settings"):
        startup_shots_count = st.slider(
            "Startup Shots to Discount",
            min_value=0, max_value=50, value=5,
            help="Number of shots after a pause to consider for warranty discount if their cycle time is bad."
        )
        stable_period_shots = st.slider(
            "Stable Production Window (shots)",
            min_value=1, max_value=100, value=10,
            help="A bad cycle is 'Scrap' if there has been no pause within this many preceding shots."
        )
        
    # --- Main Panel ---
    st.header(f"Analysis Results for {pd.to_datetime(selected_date).strftime('%d %b %Y')}")

    # Filter data based on selected date
    df_filtered = df_input[df_input['date'] == selected_date].copy()

    # Perform analysis
    analyzer = WarrantyFilter(
        df=df_filtered,
        approved_ct=approved_ct,
        pause_minutes=pause_minutes,
        ct_upper_pct=ct_upper_pct,
        ct_lower_pct=ct_lower_pct,
        startup_shots_count=startup_shots_count,
        stable_period_shots=stable_period_shots
    )
    results = analyzer.results
    
    if results and not results.get("processed_df", pd.DataFrame()).empty:
        summary_metrics = results.get("summary_metrics", {})
        processed_df = results.get("processed_df", pd.DataFrame())

        # --- Display Summary Metrics ---
        st.subheader("Summary")
        cols = st.columns(3)
        with cols[0]:
            st.metric(
                label="Total Accumulated Shots",
                value=f"{summary_metrics.get('Total Accumulated Shots', 0):,}"
            )
        with cols[1]:
            st.metric(
                label="Adjusted Shots (Affects Warranty)",
                value=f"{summary_metrics.get('Adjusted Accumulated Shots (Affects Warranty)', 0):,}",
                delta=f"-{summary_metrics.get('Discounted Startup Shots (Bad)', 0)} shots",
                delta_color="inverse",
                help="Total shots minus the bad cycles that occurred during startup."
            )
        with cols[2]:
             st.metric(
                label="Good Parts",
                value=f"{summary_metrics.get('Good Parts', 0):,}"
            )
        
        st.markdown("---")
        
        cols2 = st.columns(3)
        with cols2[0]:
            st.metric(
                label="Scrap Parts",
                value=f"{summary_metrics.get('Scrap Parts', 0):,}",
                help="Bad cycle shots that occurred during stable production."
            )
        with cols2[1]:
            st.metric(
                label="Discounted Startup Shots",
                value=f"{summary_metrics.get('Discounted Startup Shots (Bad)', 0):,}",
                help="Bad cycle shots immediately following a pause. These do not count towards warranty."
            )
        with cols2[2]:
            st.metric(
                label="Other Bad Cycles",
                value=f"{summary_metrics.get('Other Bad Cycles (Post-Pause)', 0):,}",
                help="Bad cycle shots that are not scrap and not discounted startup shots."
            )

        # --- Display Plot ---
        st.subheader("Shot Visualization")
        fig = plot_shot_analysis(processed_df, analyzer.approved_ct, analyzer.ct_upper_limit, analyzer.ct_lower_limit)
        st.plotly_chart(fig, use_container_width=True)

        # --- Display Data Table ---
        with st.expander("View Detailed Shot Data"):
            # Create a copy for display with more readable column names
            df_display = processed_df.copy()
            df_display.rename(columns={
                'shot_time': 'Shot Time',
                'actual_ct': 'Actual CT',
                'part_status': 'Part Status',
                'affects_warranty': 'Affects Warranty',
                'time_diff_minutes': 'Minutes Since Last Shot',
                'is_pause_before': 'Preceded by Pause'
            }, inplace=True)
            
            st.dataframe(df_display[[
                'Shot Time', 'Actual CT', 'Part Status', 'Affects Warranty', 
                'Minutes Since Last Shot', 'Preceded by Pause'
            ]])
    else:
        st.warning(f"No shot data found for the selected date: {pd.to_datetime(selected_date).strftime('%d %b %Y')}")

else:
    st.info("👈 Please upload an Excel file to begin the analysis.")

