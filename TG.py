Requirements
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126

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