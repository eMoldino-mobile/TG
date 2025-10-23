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
    'Bad (Post-Pause)': '#3498db',
    'Blank Shot - Excluded': '#9b59b6'
}
SHOT_TYPES = list(STATUS_COLORS.keys())

# --- Core Calculation Class ---
class WarrantyFilter:
    """
    Analyzes shot data to filter shots for warranty and end-of-life purposes.
    It expects a DataFrame with standardized column names (lowercase, snake_case).
    """
    def __init__(self, df: pd.DataFrame, approved_ct: float, pause_minutes: int,
                 ct_upper_pct: float, ct_lower_pct: float, startup_shots_count: int,
                 stable_period_shots: int, blank_shot_upper_dev_pct: float, 
                 blank_shot_lower_dev_pct: float, shot_config: dict):
        
        # Carry over all original columns
        self.df_raw = df.copy() 
        self.approved_ct = approved_ct
        self.pause_minutes = pause_minutes
        self.ct_upper_limit = approved_ct * (1 + ct_upper_pct / 100)
        self.ct_lower_limit = approved_ct * (1 - ct_lower_pct / 100)
        self.startup_shots_count = startup_shots_count
        self.stable_period_shots = stable_period_shots
        self.blank_shot_upper_threshold = approved_ct * (1 + blank_shot_upper_dev_pct / 100)
        self.blank_shot_lower_threshold = approved_ct * (1 - blank_shot_lower_dev_pct / 100)
        self.shot_config = shot_config 

    def _prepare_data(self) -> pd.DataFrame:
        """Prepares the raw DataFrame for analysis."""
        df = self.df_raw.copy()
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

    def analyze_shots(self) -> dict:
        """
        Runs the full analysis, classification, and configuration logic.
        This is now separated from __init__.
        """
        df = self._prepare_data()
        if df.empty:
            return {}

        # --- Steps 1-5: CLASSIFICATION ---
        # This logic now *only* sets the 'part_status' string.
        
        # 1. Default classification
        df['part_status'] = 'Good'
        
        # 2. Flag all bad cycles initially
        is_bad_cycle = (df['actual_ct'] > self.ct_upper_limit) | (df['actual_ct'] < self.ct_lower_limit)
        df.loc[is_bad_cycle, 'part_status'] = 'Bad (Post-Pause)'

        # 3. Identify Blank Shots (Highest Priority)
        is_blank_shot = (df['actual_ct'] > self.blank_shot_upper_threshold) | (df['actual_ct'] < self.blank_shot_lower_threshold)
        df.loc[is_blank_shot, 'part_status'] = 'Blank Shot - Excluded'
        
        # 4. Identify and re-classify Startup Shots
        df['is_pause_before'] = df['time_diff_minutes'] > self.pause_minutes
        pause_indices = df[df['is_pause_before']].index

        for idx in pause_indices:
            startup_range_start = idx 
            startup_range_end = min(idx + self.startup_shots_count, len(df)) 
            
            for startup_idx in range(startup_range_start, startup_range_end):
                # We only re-classify shots that are currently 'Bad (Post-Pause)'
                if df.loc[startup_idx, 'part_status'] == 'Bad (Post-Pause)':
                    df.loc[startup_idx, 'part_status'] = 'Startup - Bad'

        # 5. Identify Scrap Shots from the remaining bad cycles
        # A shot is scrap if it's still 'Bad (Post-Pause)' and occurs in a stable window.
        df['pause_in_window'] = df['is_pause_before'].rolling(window=self.stable_period_shots, min_periods=1).max()
        is_stable = df['pause_in_window'] != 1.0
        
        is_scrap_classification = (df['part_status'] == 'Bad (Post-Pause)') & is_stable
        df.loc[is_scrap_classification, 'part_status'] = 'Scrap'
        
        
        # --- Step 6: CONFIGURATION (Apply Consequences) ---
        # Apply the user-defined settings based on the classification
        
        is_scrap_map = {status: props['is_scrap'] for status, props in self.shot_config.items()}
        affects_warranty_map = {status: props['affects_warranty'] for status, props in self.shot_config.items()}

        df['is_scrap'] = df['part_status'].map(is_scrap_map).fillna(False).astype(bool)
        df['affects_warranty'] = df['part_status'].map(affects_warranty_map).fillna(False).astype(bool)


        # --- Summarize Results ---
        total_shots = len(df)
        summary = {
            # Final outcomes based on configuration
            "Total Accumulated Shots": total_shots,
            "Adjusted Accumulated Shots (Affects Warranty)": int(df['affects_warranty'].sum()),
            "Total Scrap Parts (from Config)": int(df['is_scrap'].sum()),
            
            # Raw classification counts
            "Good": (df['part_status'] == 'Good').sum(),
            "Scrap": (df['part_status'] == 'Scrap').sum(), # Note: This is the *classification*
            "Startup - Bad": (df['part_status'] == 'Startup - Bad').sum(),
            "Bad (Post-Pause)": (df['part_status'] == 'Bad (Post-Pause)').sum(),
            "Blank Shot - Excluded": (df['part_status'] == 'Blank Shot - Excluded').sum()
        }
        
        return {"processed_df": df, "summary_metrics": summary}

# --- Plotting Function ---
def plot_shot_analysis(df, approved_ct, upper_limit, lower_limit):
    if df.empty:
        st.info("No data to display for the selected date.")
        return go.Figure()

    fig = go.Figure()

    df['color'] = df['part_status'].map(STATUS_COLORS)
    fig.add_trace(go.Bar(
        x=df['shot_time'],
        y=df['actual_ct'],
        marker_color=df['color'],
        name='Cycle Time',
        showlegend=False,
        # Add customdata for hover
        customdata=np.stack((
            df['part_status'], 
            df['is_scrap'], 
            df['affects_warranty']
        ), axis=-1),
        hovertemplate=(
            '<b>Shot Time</b>: %{x}<br>'
            '<b>Actual CT</b>: %{y:.2f}s<br>'
            '<b>Classification</b>: %{customdata[0]}<br>'
            '<b>Is Scrap?</b>: %{customdata[1]}<br>'
            '<b>Affects Warranty?</b>: %{customdata[2]}'
            '<extra></extra>' # Hides the trace name
        )
    ))

    for status, color in STATUS_COLORS.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color, name=status, showlegend=True))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='purple', dash='dash', width=1.5), # <-- FIX was here
        name='Production Pause', showlegend=True
    ))

    fig.add_hline(y=approved_ct, line_dash="dot", line_color="grey",
                  annotation_text="Approved CT", annotation_position="bottom right")
    fig.add_hline(y=upper_limit, line_dash="dash", line_color="orange",
                  annotation_text="Upper Tolerance", annotation_position="top right")
    fig.add_hline(y=lower_limit, line_dash="dash", line_color="orange",
                  annotation_text="Lower Tolerance", annotation_position="bottom right")
    
    pause_times = df[df['is_pause_before']]['shot_time']
    for p_time in pause_times:
        fig.add_vline(x=p_time, line_width=1.5, line_dash="dash", line_color="purple")

    y_axis_max = max(df['actual_ct'].max() * 1.1, upper_limit * 1.5) if not df.empty else approved_ct * 2
    
    fig.update_layout(
        title="Shot-by-Shot Cycle Time Analysis (Hover for Details)",
        xaxis_title="Shot Time",
        yaxis_title="Actual Cycle Time (seconds)",
        yaxis_range=[0, y_axis_max],
        legend_title_text='Legend',
        bargap=0.1,
        hovermode="x unified"
    )
    return fig

# --- Main App UI ---
st.title("Warranty & End-of-Life (EOL) Shot Filter")

st.sidebar.header("⚙️ Analysis Controls")
uploaded_file = st.sidebar.file_uploader("Upload your shot data Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    # Store original columns before standardizing
    original_columns = list(df_input.columns) 
    df_input.columns = [col.strip().lower().replace(' ', '_') for col in df_input.columns]

    if 'approved_ct' not in df_input.columns or df_input['approved_ct'].dropna().empty:
        st.error("Error: The uploaded Excel file must contain a column named 'APPROVED CT' with at least one valid number.")
        st.stop()

    st.sidebar.success(f"File '{uploaded_file.name}' loaded successfully!")
    
    df_input['shot_time'] = pd.to_datetime(df_input['shot_time'], errors='coerce')
    df_input.dropna(subset=['shot_time'], inplace=True)
    df_input['date'] = df_input['shot_time'].dt.date
    available_dates = sorted(df_input["date"].unique())
    
    selected_date = st.sidebar.selectbox(
        "Select Date to Analyze",
        options=available_dates,
        index=len(available_dates) - 1,
        format_func=lambda d: pd.to_datetime(d).strftime('%d %b %Y')
    )

    st.sidebar.markdown("---")
    
    default_approved_ct = df_input['approved_ct'].dropna().iloc[0]
    st.sidebar.info(f"Using 'Approved CT' from file: {default_approved_ct:.2f}s")

    approved_ct = st.sidebar.number_input(
        "Approved Cycle Time (seconds)",
        min_value=0.1, value=float(default_approved_ct), step=0.1,
        help="This value is automatically populated from the 'APPROVED CT' column in your file."
    )
    
    with st.sidebar.expander("Threshold Settings", expanded=True):
        pause_minutes = st.slider("Production Pause Threshold (minutes)", 1, 480, 30)
        ct_upper_pct = st.slider("Cycle Time Upper Tolerance (%)", 1, 100, 15)
        ct_lower_pct = st.slider("Cycle Time Lower Tolerance (%)", 1, 100, 15)

    with st.sidebar.expander("Warranty & Scrap Logic Settings"):
        blank_shot_upper_dev_pct = st.slider(
            "Blank Shot Upper Deviation (%)",
            min_value=1, max_value=1000, value=400,
            help="Defines the % increase from 'Approved CT' to set the upper blank shot limit."
        )
        blank_shot_upper_seconds = approved_ct * (1 + blank_shot_upper_dev_pct / 100)
        st.sidebar.markdown(f"> _Corresponds to: **{blank_shot_upper_seconds:.2f} seconds**_")

        blank_shot_lower_dev_pct = st.slider(
            "Blank Shot Lower Deviation (%)",
            min_value=1, max_value=99, value=80,
            help="Defines the % decrease from 'Approved CT' to set the lower blank shot limit."
        )
        blank_shot_lower_seconds = approved_ct * (1 - blank_shot_lower_dev_pct / 100)
        st.sidebar.markdown(f"> _Corresponds to: **{blank_shot_lower_seconds:.2f} seconds**_")
        
        # --- FIX IS HERE ---
        startup_shots_count = st.slider(
            "Startup Shot Window (shots)", 0, 50, 5,
            help="The number of shots to check *after* a pause. Any bad shots within this window will be classified as 'Startup - Bad'."
        )
        # --- END OF FIX ---
        stable_period_shots = st.slider("Stable Production Window (shots)", 1, 100, 10)
        
    st.sidebar.markdown("---")
    
    # --- NEW: Shot Type Configuration ---
    # Set default configurations based on the *original* logic
    DEFAULT_CONFIGS = {
        'Good': {'is_scrap': False, 'affects_warranty': True},
        'Scrap': {'is_scrap': True, 'affects_warranty': True},
        'Startup - Bad': {'is_scrap': False, 'affects_warranty': False},
        'Bad (Post-Pause)': {'is_scrap': False, 'affects_warranty': True},
        'Blank Shot - Excluded': {'is_scrap': False, 'affects_warranty': False}
    }

    shot_config = {}
    with st.sidebar.expander("Shot Type Property Configuration", expanded=True):
        st.caption("Configure the final properties for each classified shot type.")
        for shot_type in SHOT_TYPES:
            st.markdown(f"**{shot_type}**")
            cols = st.columns(2)
            default_props = DEFAULT_CONFIGS[shot_type]
            is_scrap = cols[0].checkbox(
                "Is Scrap?", 
                default_props['is_scrap'], 
                key=f"{shot_type}_scrap",
                help=f"Mark '{shot_type}' shots as scrap?"
            )
            affects_warranty = cols[1].checkbox(
                "Affects Warranty?", 
                default_props['affects_warranty'], 
                key=f"{shot_type}_warranty",
                help=f"Count '{shot_type}' shots towards warranty?"
            )
            shot_config[shot_type] = {
                'is_scrap': is_scrap, 
                'affects_warranty': affects_warranty
            }
    
    # --- End of New Config Section ---
    
    st.header(f"Analysis Results for {pd.to_datetime(selected_date).strftime('%d %b %Y')}")

    # We pass the original df_input columns to the filter
    # The filter class will only use what it needs (shot_time, actual_ct)
    # but it will carry over the rest of the columns
    df_filtered = df_input[df_input['date'] == selected_date].copy()

    # Instantiate the analyzer with the new config
    analyzer = WarrantyFilter(
        df=df_filtered,
        approved_ct=approved_ct,
        pause_minutes=pause_minutes,
        ct_upper_pct=ct_upper_pct,
        ct_lower_pct=ct_lower_pct,
        startup_shots_count=startup_shots_count,
        stable_period_shots=stable_period_shots,
        blank_shot_upper_dev_pct=blank_shot_upper_dev_pct,
        blank_shot_lower_dev_pct=blank_shot_lower_dev_pct,
        shot_config=shot_config  # Pass the new config
    )
    # Run the analysis
    results = analyzer.analyze_shots()
    
    if results and not results.get("processed_df", pd.DataFrame()).empty:
        summary_metrics = results.get("summary_metrics", {})
        processed_df = results.get("processed_df", pd.DataFrame())

        st.subheader("Summary (Based on Configuration)")
        cols = st.columns(3)
        cols[0].metric("Total Accumulated Shots", f"{summary_metrics.get('Total Accumulated Shots', 0):,}")
        cols[1].metric("Adjusted Shots (Affects Warranty)", f"{summary_metrics.get('Adjusted Accumulated Shots (Affects Warranty)', 0):,}")
        cols[2].metric("Total Scrap Parts (from Config)", f"{summary_metrics.get('Total Scrap Parts (from Config)', 0):,}")
        
        st.markdown("---")
        
        st.subheader("Shot Classification Counts")
        st.caption("These are the raw counts *before* your configuration is applied.")
        cols2 = st.columns(5)
        cols2[0].metric("Good", f"{summary_metrics.get('Good', 0):,}")
        cols2[1].metric("Scrap (Classified)", f"{summary_metrics.get('Scrap', 0):,}")
        cols2[2].metric("Startup - Bad", f"{summary_metrics.get('Startup - Bad', 0):,}")
        cols2[3].metric("Bad (Post-Pause)", f"{summary_metrics.get('Bad (Post-Pause)', 0):,}")
        cols2[4].metric("Blank Shot - Excluded", f"{summary_metrics.get('Blank Shot - Excluded', 0):,}")

        st.subheader("Shot Visualization")
        fig = plot_shot_analysis(processed_df, analyzer.approved_ct, analyzer.ct_upper_limit, analyzer.ct_lower_limit)
        st.plotly_chart(fig, use_container_width=True)

        # --- FIX IS HERE ---
        with st.expander("View Detailed Shot Data"):
            df_display = processed_df.copy()
            
            # Define the 7 columns we are renaming
            rename_map = {
                'shot_time': 'Shot Time', 'actual_ct': 'Actual CT',
                'part_status': 'Classification', 
                'is_scrap': 'Is Scrap? (Config)', 
                'affects_warranty': 'Affects Warranty? (Config)',
                'time_diff_minutes': 'Minutes Since Last Shot', 
                'is_pause_before': 'Preceded by Pause'
            }
            df_display.rename(columns=rename_map, inplace=True)
            
            # These are the 7 columns we want to show first, in order
            pretty_cols_ordered = list(rename_map.values())
            
            # These are the original (snake_case) names of the columns we renamed
            internal_cols_renamed = list(rename_map.keys())
            
            # Find all *other* columns from the original file that weren't renamed
            # and aren't the date/ct helpers.
            # We check against df_input.columns to get all original columns.
            other_original_cols = [
                col for col in df_input.columns 
                if col not in internal_cols_renamed and col not in ['date', 'approved_ct']
            ]
            
            # Combine the lists and display
            # This ensures all original data (like 'operator_name', etc.)
            # that was in df_input is also shown here.
            
            # Final check: only include other_original_cols if they actually
            # exist in the final df_display (which they should, as they're
            # carried over from df_raw)
            final_other_cols = [col for col in other_original_cols if col in df_display.columns]
            
            st.dataframe(df_display[pretty_cols_ordered + final_other_cols])
        # --- END OF FIX ---
            
    else:
        st.warning(f"No shot data found for the selected date: {pd.to_datetime(selected_date).strftime('%d %b %Y')}")

else:
    st.info("👈 Please upload an Excel file to begin the analysis.")

