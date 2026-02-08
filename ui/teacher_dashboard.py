import sys
import os
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
import streamlit as st
import pandas as pd
import joblib
import base64
from datetime import datetime
import plotly.express as px
import streamlit.components.v1 as components

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Internal Project Imports
from src.data.preprocessing import (
    add_student_id,
    drop_unused_columns,
    encode_ordinal_features,
    encode_binary_features,
)
from src.features.effort_gap import compute_effort_outcome_gap
from src.features.resource_mismatch import (
    compute_resource_index,
    add_resource_mismatch_flag,
)
from src.features.persona_clustering import (
    prepare_clustering_features,
    assign_persona_clusters,
    map_failure_mode_persona,
)
from src.features.primary_lever import add_primary_lever
from src.features.intervention_simulation import (
    add_expected_score_improvement,
)
from src.explainability.build_payload import build_genai_payload
from src.explainability.genai_engine import generate_teacher_explanation

from ui.visuals import plot_risk_distribution, plot_priority_scatter, plot_student_radar

# ----------------------------
# CUSTOM CSS & STYLING
# ----------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    /* Base styling with your color palette */
    :root {
        --page-bg: #000E24;
        --card-bg: #001433;
        --primary: #0066FF;
        --primary-hover: #0052CC;
        --text-light: #E5F0FF;
        --text-muted: #8CA3C7;
        --success: #10B981;
        --danger: #FF4B4B;
        --warning: #F59E0B;
        --border: rgba(0, 102, 255, 0.2);
        --glass-bg: rgba(255, 255, 255, 0.05);
    }
    
    /* Main page styling */
    .stApp {
        background: linear-gradient(135deg, var(--page-bg) 0%, #001A3D 100%);
        color: var(--text-light);
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 102, 255, 0.15);
        border-color: rgba(0, 102, 255, 0.4);
    }
    
    /* Bento grid layout */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(12, 1fr);
        grid-auto-rows: minmax(100px, auto);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .bento-1 { grid-column: span 4; grid-row: span 1; }
    .bento-2 { grid-column: span 4; grid-row: span 1; }
    .bento-3 { grid-column: span 4; grid-row: span 1; }
    .bento-4 { grid-column: span 8; grid-row: span 2; }
    .bento-5 { grid-column: span 4; grid-row: span 2; }
    
    /* Custom metric cards */
    .metric-card {
        background: var(--card-bg);
        border-radius: 16px;
        padding: 1.5rem;
        border-left: 4px solid var(--primary);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--primary), #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: var(--text-muted);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-high-risk { background: rgba(255, 75, 75, 0.15); color: var(--danger); border: 1px solid var(--danger); }
    .status-medium-risk { background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid var(--warning); }
    .status-low-risk { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid var(--success); }
    
    /* Persona tags */
    .persona-tag {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
    
    /* Custom buttons */
    .stButton button {
        background: linear-gradient(90deg, var(--primary), #3B82F6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 102, 255, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg { background-color: rgba(0, 20, 51, 0.95) !important; }
    
    /* Dataframe styling */
    .dataframe {
        background: var(--card-bg) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .dataframe th {
        background: rgba(0, 102, 255, 0.1) !important;
        color: var(--text-light) !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        color: var(--text-muted) !important;
    }
    
    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 2rem 0;
    }
    
    /* Progress bars */
    .progress-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary), #3B82F6);
        border-radius: 10px;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-hover);
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def get_risk_badge(z_score):
    if z_score <= -0.9:
        return '<span class="status-badge status-high-risk">🚨 High Risk</span>'
    elif z_score <= -0.5:
        return '<span class="status-badge status-medium-risk">⚠️ Medium Risk</span>'
    else:
        return '<span class="status-badge status-low-risk">✅ Low Risk</span>'

def get_persona_color(persona):
    color_map = {
        "Overworked Struggler": "linear-gradient(135deg, #8B5CF6, #EC4899)",
        "Disengaged Despite Resources": "linear-gradient(135deg, #10B981, #059669)",
        "Resource-Constrained Achiever": "linear-gradient(135deg, #3B82F6, #1D4ED8)",
        "Balanced Performer": "linear-gradient(135deg, #F59E0B, #D97706)",
    }
    return color_map.get(persona, "var(--primary)")

def create_download_link(df, filename="student_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">📥 Download {filename}</a>'
    return href

# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(
    page_title="PARIX | Student Intervention Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
inject_custom_css()

load_dotenv()

# ----------------------------
# LOAD ARTIFACTS (Cached)
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.getenv("MODEL_PATH", "models/exam_model.joblib"))
    scaler = joblib.load(os.getenv("SCALER_PATH", "models/scaler.joblib"))
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv(os.getenv("DATA_PATH", "notebooks/Student_data.csv"))

try:
    model, scaler = load_artifacts()
    df_raw = load_data()
except Exception as e:
    st.error(f"❌ Error loading data or models: {e}")
    st.stop()

# ----------------------------
# PIPELINE
# ----------------------------
@st.cache_data
def build_feature_table(df):
    df = add_student_id(df)
    
    # 1. Preprocessing
    df = drop_unused_columns(df)
    df = encode_ordinal_features(df)
    df = encode_binary_features(df)

   # Define feature columns for model input
    feature_cols = [
        "Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", 
        "Tutoring_Sessions", "Physical_Activity", "Internet_Access", 
        "Extracurricular_Activities", "Learning_Disabilities", "Gender", 
        "School_Type_Public", "Parental_Involvement", "Access_to_Resources", 
        "Motivation_Level", "Family_Income", "Peer_Influence"
    ]

    # 2. Effort Gap Analysis
    df = compute_effort_outcome_gap(df, model, scaler, feature_cols)

    # 3. Resource Mismatch
    resource_cols = ["Access_to_Resources", "Internet_Access", "Family_Income"]
    df, _ = compute_resource_index(df, resource_cols)
    df = add_resource_mismatch_flag(df)

    # 4. Persona Clustering
    cluster_features = ["gap_for_clustering", "Sleep_Hours", "Motivation_Level", "Attendance", "resource_index"]
    X_cluster_scaled, _ = prepare_clustering_features(df, cluster_features)
    df, _ = assign_persona_clusters(df, X_cluster_scaled, n_clusters=4)
    df = map_failure_mode_persona(df)

    # 5. Interventions
    df = add_primary_lever(df)
    df = add_expected_score_improvement(df, model, scaler, feature_cols)

    return df

if "df" not in st.session_state:
    st.session_state.df = build_feature_table(df_raw)

df = st.session_state.df

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="background: linear-gradient(90deg, #0066FF, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">🎓 PARIX</h1>
        <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0.5rem 0;">Student Intervention Analytics</p>
        <div style="height: 1px; background: linear-gradient(90deg, transparent, var(--border), transparent); margin: 1rem 0;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    screen = st.radio(
        "Dashboard Views",
        ["📊 Weekly Priority", "🔍 Student Deep Dive", "📈 Class Insights"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### 📅 Week Overview")
    week = st.selectbox("Select Week", ["Week 1", "Week 2", "Week 3", "Week 4", "Current Week"])
    
    st.markdown("### 👥 Filter Students")
    risk_filter = st.multiselect(
        "Risk Level",
        ["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk"]
    )
    
    persona_filter = st.multiselect(
        "Persona Types",
        df["failure_mode_persona"].unique().tolist(),
        default=df["failure_mode_persona"].unique().tolist()
    )
    
    st.markdown("---")
    
    # Export functionality
    st.markdown("### 📤 Export Data")
    if st.button("📥 Export Current View as CSV"):
        filtered_df = df.copy()
        if risk_filter:
            risk_map = {'High Risk': -0.9, 'Medium Risk': -0.5, 'Low Risk': 0}
            filtered_df = filtered_df[filtered_df['effort_outcome_gap_z'].apply(
                lambda x: any(x <= risk_map[r] for r in risk_filter)
            )]
        if persona_filter:
            filtered_df = filtered_df[filtered_df['failure_mode_persona'].isin(persona_filter)]
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"parix_student_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    


# ----------------------------
# MAIN CONTENT
# ----------------------------
if screen == "📊 Weekly Priority":
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("""
        <h1 style="margin-bottom: 0;">📊 Weekly Priority Matrix</h1>
        <p style="color: var(--text-muted); margin-top: 0;">Identifying students needing immediate attention</p>
        """, unsafe_allow_html=True)
    with col2:
        st.metric("Total Students", len(df))
    with col3:
        st.metric("Week", week)
    
    # Bento Grid - Top Metrics
    st.markdown('<div class="bento-grid">', unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="glass-card bento-1">
                <div class="metric-label">🚨 High Risk Students</div>
                <div class="metric-value">{len(df[df["effort_outcome_gap_z"] <= -0.9])}</div>
                <div style="color: var(--danger); font-size: 0.9rem;">Need immediate intervention</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="glass-card bento-2">
                <div class="metric-label">📈 Avg. Potential Gain</div>
                <div class="metric-value">+{df['expected_score_improvement'].mean():.1f}</div>
                <div style="color: var(--success); font-size: 0.9rem;">With targeted interventions</div>
            </div>
            """, unsafe_allow_html=True)
        counts = df['primary_lever'].value_counts()
        second_most = counts.index[1] if len(counts) > 1 else "N/A"
        with col3:
            st.markdown(f"""
            <div class="glass-card bento-3">
                <div class="metric-label">🎯 Primary Intervention</div>
                <div style="font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0; color: var(--primary);">
                    {second_most}
                </div>
                <div style="color: var(--text-muted); font-size: 0.9rem;">Most common leverage point</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk Distribution Chart
    st.markdown("### 📋 Risk Distribution Overview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = plot_risk_distribution(df)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E5F0FF'),
            legend=dict(font=dict(color='#8CA3C7'))
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="height: 300px;">
            <h4 style="margin-top: 0;">🎯 Action Plan</h4>
            <div style="margin: 1rem 0;">
                <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                    <div style="width: 12px; height: 12px; background: #EF553B; border-radius: 50%; margin-right: 0.5rem;"></div>
                    <span>High Risk: Schedule 1-on-1 meetings</span>
                </div>
                <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                    <div style="width: 12px; height: 12px; background: #FECB52; border-radius: 50%; margin-right: 0.5rem;"></div>
                    <span>Medium Risk: Group workshops</span>
                </div>
                <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                    <div style="width: 12px; height: 12px; background: #00CC96; border-radius: 50%; margin-right: 0.5rem;"></div>
                    <span>Low Risk: Monitor & encourage</span>
                </div>
            </div>
            <div class="custom-divider"></div>
            <p style="font-size: 0.9rem; color: var(--text-muted);">
            <strong>Tip:</strong> Focus on top 3 high-risk students this week for maximum impact.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Priority Matrix
    st.markdown("### 🎯 Student Priority Matrix")
    fig = plot_priority_scatter(df)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E5F0FF'),
        title_font=dict(size=20, color='#E5F0FF')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Priority Students Table
    st.markdown("### 👥 Top  Priority Students")
    priority_df = df.sort_values("effort_outcome_gap_z").head(15).copy()
    
    # Add risk badges to dataframe
    priority_df["Risk"] = priority_df["effort_outcome_gap_z"].apply(
        lambda x: "🚨 High" if x <= -0.9 else ("⚠️ Medium" if x <= -0.5 else "✅ Low")
    )
    
    display_cols = {
        "Student_ID": "Student ID",
        "Risk": "Risk Level",
        "failure_mode_persona": "Persona",
        "primary_lever": "Primary Lever",
        "expected_score_improvement": "Potential Gain"
    }
    
    st.dataframe(
        priority_df[list(display_cols.keys())].rename(columns=display_cols),
        use_container_width=True,
        height=400
    )

elif screen == "🔍 Student Deep Dive":
    st.title("🔍 Student Deep Dive Analysis")
    st.caption("Individual student analytics and personalized intervention strategies")
    
    # Student Selection with Comparison Option
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_id = st.selectbox(
            "Select Student ID",
            df["Student_ID"].unique(),
            format_func=lambda x: f"Student {x}"
        )
    
    with col2:
        compare_mode = st.checkbox("Compare with another student")
    
    with col3:
        if compare_mode:
            compare_id = st.selectbox(
                "Compare with",
                df[df["Student_ID"] != selected_id]["Student_ID"].unique(),
                format_func=lambda x: f"Student {x}"
            )
    
    student_row = df[df["Student_ID"] == selected_id].iloc[0]
    
    if compare_mode and 'compare_id' in locals():
        compare_row = df[df["Student_ID"] == compare_id].iloc[0]
    
    # Student Overview Cards
    st.subheader("📋 Student Overview")
    
    if compare_mode and 'compare_row' in locals():

        # COMPARISON VIEW
        col1, col2 = st.columns(2)
        
        for idx, (col, row) in enumerate(zip([col1, col2], [student_row, compare_row])):
            with col:
                # Create a container for the student card
                student_card = st.container()
                with student_card:
                    # Header section
                    header_col1, header_col2 = st.columns([2, 1])
                    with header_col1:
                        st.markdown(f"### Student {row['Student_ID']}")
                    with header_col2:
                        risk_badge = get_risk_badge(row["effort_outcome_gap_z"])
                        st.markdown(risk_badge, unsafe_allow_html=True)
                    
                    # Divider
                    st.markdown("---")
                    
                    # Persona section
                    persona_col1, persona_col2 = st.columns([1, 2])
                    with persona_col1:
                        st.markdown("**Persona:**")
                    with persona_col2:
                        persona_color = get_persona_color(row["failure_mode_persona"])
                        st.markdown(
                            f'<span style="background-color: {persona_color}; padding: 4px 12px; border-radius: 12px; color: white; font-weight: bold;">{row["failure_mode_persona"]}</span>',
                            unsafe_allow_html=True
                        )
                    
                    # Primary Lever
                    lever_col1, lever_col2 = st.columns([1, 2])
                    with lever_col1:
                        st.markdown("**Primary Lever:**")
                    with lever_col2:
                        st.markdown(f'<span style="color: #0066FF; font-weight: bold;">{row["primary_lever"]}</span>', unsafe_allow_html=True)
                    
                    # Potential Gain
                    gain_col1, gain_col2 = st.columns([1, 2])
                    with gain_col1:
                        st.markdown("**Potential Gain:**")
                    with gain_col2:
                        st.markdown(f'<span style="color: #10B981; font-weight: bold;">+{row["expected_score_improvement"]:.2f}</span>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Metrics Grid - Using columns for clean layout
                    st.markdown("**Key Metrics:**")
                    metrics_cols = st.columns(4)
                    
                    metrics_data = [
                        ("Attendance", f"{row['Attendance']}%"),
                        ("Sleep Hours", row['Sleep_Hours']),
                        ("Motivation", f"{row['Motivation_Level']}/10"),
                        ("Study Hours", row['Hours_Studied'])
                    ]
                    
                    for i, (metric_name, metric_value) in enumerate(metrics_data):
                        with metrics_cols[i]:
                            st.markdown(
                                f'<div style="text-align: center; padding: 10px; background: rgba(0, 102, 255, 0.05); border-radius: 8px; border: 1px solid rgba(0, 102, 255, 0.1);">'
                                f'<div style="font-size: 18px; font-weight: bold; color: #E5F0FF;">{metric_value}</div>'
                                f'<div style="font-size: 12px; color: #8CA3C7;">{metric_name}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
    
    


        
        with col2:
            # Radar Chart
            st.subheader("📊 Performance vs Class Average")
            fig = plot_student_radar(student_row, df)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E5F0FF'),
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        color='#8CA3C7'
                    ),
                    angularaxis=dict(color='#8CA3C7')
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional stats in a clean card
            with st.container():
                st.markdown(
                    '<div style="padding: 15px; background: rgba(0, 20, 51, 0.5); border-radius: 12px; border: 1px solid rgba(0, 102, 255, 0.2); margin-top: 20px;">'
                    '<div style="font-size: 14px; color: #8CA3C7; margin-bottom: 10px;"><b>📊 Additional Stats</b></div>'
                    '<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">'
                    '<span style="color: #8CA3C7;">Previous Scores:</span>'
                    f'<span style="color: #E5F0FF; font-weight: bold;">{student_row["Previous_Scores"]}</span>'
                    '</div>'
                    '<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">'
                    '<span style="color: #8CA3C7;">Tutoring Sessions:</span>'
                    f'<span style="color: #E5F0FF; font-weight: bold;">{student_row["Tutoring_Sessions"]}</span>'
                    '</div>'
                    '<div style="display: flex; justify-content: space-between;">'
                    '<span style="color: #8CA3C7;">Resource Index:</span>'
                    f'<span style="color: #E5F0FF; font-weight: bold;">{student_row["resource_index"]:.2f}</span>'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )
    
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            components.html(
                f"""
                <style>
                    .card {{
                        background:#0e1117;
                        border:1px solid #1f2937;
                        border-radius:16px;
                        padding:18px;
                        color:white;
                        font-family:sans-serif;
                    }}
                    .grid-3 {{
                        display:grid;
                        grid-template-columns:repeat(3,1fr);
                        gap:14px;
                    }}
                    .grid-2 {{
                        display:grid;
                        grid-template-columns:repeat(2,1fr);
                        gap:14px;
                    }}
                    .metric {{
                        padding:14px;
                        border-radius:12px;
                        background:rgba(59,130,246,.15);
                        border-left:4px solid #3b82f6;
                    }}
                    .green {{ background:rgba(16,185,129,.15); border-left-color:#10b981; }}
                    .yellow {{ background:rgba(245,158,11,.15); border-left-color:#f59e0b; }}
                    .red {{ background:rgba(239,68,68,.15); border-left-color:#ef4444; }}
                    .center {{ text-align:center; }}
                    .label {{ font-weight:600; opacity:.9; }}
                    .value {{ font-size:22px; font-weight:700; margin-top:6px; }}
                    .bar {{
                        height:8px;
                        background:#1f2937;
                        border-radius:4px;
                        overflow:hidden;
                        margin-top:8px;
                    }}
                    .fill {{
                        height:100%;
                        background:#3b82f6;
                    }}
                </style>

                <div class="card">

                    <!-- HEADER -->
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h3>👤 Student {student_row['Student_ID']}</h3>
                        <div style="padding:10px 14px; background:#2563eb; border-radius:12px;">
                            🎭 {student_row['failure_mode_persona']}
                        </div>
                    </div>

                    <hr style="border:0; border-top:1px solid #1f2937; margin:16px 0;">

                    <!-- KEY INSIGHTS -->
                    <div class="grid-3">
                        <div class="metric">
                            <div class="label">🎯 Primary Lever</div>
                            <div class="value">{student_row['primary_lever']}</div>
                        </div>

                        <div class="metric green">
                            <div class="label">📈 Potential Gain</div>
                            <div class="value">+{student_row['expected_score_improvement']:.1f}</div>
                        </div>

                        <div class="metric red">
                            <div class="label">📊 Risk Score</div>
                            <div class="value">{student_row['effort_outcome_gap_z']:.2f}</div>
                        </div>
                    </div>

                    <hr style="border:0; border-top:1px solid #1f2937; margin:16px 0;">

                    <!-- PERFORMANCE -->
                    <div class="grid-2">

                        <div class="metric center">
                            <div class="label">🎯 Attendance</div>
                            <div class="value">{student_row['Attendance']}%</div>
                            <div class="bar">
                                <div class="fill" style="width:{student_row['Attendance']}%;"></div>
                            </div>
                        </div>

                        <div class="metric center" style="background:rgba(139,92,246,.15); border-left-color:#8b5cf6;">
                            <div class="label">😴 Sleep</div>
                            <div class="value">{student_row['Sleep_Hours']}h</div>
                        </div>

                        <div class="metric center green">
                            <div class="label">🚀 Motivation</div>
                            <div class="value">{student_row['Motivation_Level']}/2</div>
                        </div>

                        <div class="metric center yellow">
                            <div class="label">📚 Study Hours</div>
                            <div class="value">{student_row['Hours_Studied']}h</div>
                            <div style="font-size:13px; opacity:.7;">
                                {student_row['Hours_Studied']/4:.1f}h / week
                            </div>
                        </div>

                    </div>

                </div>
                """,
                height=920
            )
        with col2:
            # Radar Chart Section
            st.markdown("### 📊 Performance Radar")
            
            # Create container for radar chart
            radar_container = st.container()
            with radar_container:
                fig = plot_student_radar(student_row, df)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E5F0FF'),
                    height=300,
                    margin=dict(t=40, b=40, l=40, r=40)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional Stats Card
            st.markdown("### 📋 Additional Statistics")
            
            stats_container = st.container()
            with stats_container:
                # Using columns for clean stat display
                stat1, stat2 = st.columns(2)
                
                with stat1:
                    st.metric(
                        label="Previous Scores",
                        value=f"{student_row['Previous_Scores']}",
                        delta=None
                    )
                
                with stat2:
                    st.metric(
                        label="Tutoring Sessions",
                        value=f"{student_row['Tutoring_Sessions']}",
                        delta=None
                    )
                
                # Resource Index with visual indicator
                resource_index = student_row['resource_index']
                st.markdown("**Resource Index:**")
                
                # Visual bar for resource index
                st.markdown(
                    f'<div style="display: flex; align-items: center; gap: 10px; margin: 10px 0;">'
                    f'<div style="flex: 1; height: 8px; background: rgba(140, 163, 199, 0.2); border-radius: 4px; overflow: hidden;">'
                    f'<div style="width: {resource_index * 100}%; height: 100%; background: linear-gradient(90deg, #0066FF, #3B82F6);"></div>'
                    f'</div>'
                    f'<span style="color: #E5F0FF; font-weight: bold; min-width: 40px;">{resource_index:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Resource mismatch flag
                mismatch = student_row.get('resource_mismatch_flag', 'LOW')
                mismatch_colors = {
                    'HIGH': ('#FF4B4B', '⚠️ High Mismatch'),
                    'MEDIUM': ('#F59E0B', '📊 Medium Mismatch'),
                    'LOW': ('#10B981', '✅ Low Mismatch')
                }
                mismatch_color, mismatch_text = mismatch_colors.get(mismatch, ('#8CA3C7', 'Unknown'))
                
                st.markdown(
                    f'<div style="padding: 10px; background: {mismatch_color}20; '
                    f'border-radius: 8px; border: 1px solid {mismatch_color}40; '
                    f'text-align: center; margin-top: 10px;">'
                    f'<div style="color: {mismatch_color}; font-weight: bold;">{mismatch_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # AI-Generated Insights Section
    st.markdown("---")
    st.subheader("🤖 AI-Generated Intervention Strategy")
    
    ai_col1, ai_col2 = st.columns([3, 1])
    with ai_col2:
        if st.button("🎯 Generate Playbook", type="primary", use_container_width=True):
            st.session_state.generate_playbook = True
    
    if st.session_state.get('generate_playbook', False):
        with st.spinner("🔍 Analyzing student patterns..."):
            try:
                input_row = student_row.to_dict()
                input_row["primary_intervention_lever"] = student_row["primary_lever"]
                input_row["effort_outcome_gap"] = student_row["effort_outcome_gap"]
                input_row["persona_label"] = student_row["failure_mode_persona"]
                input_row["student_context"] = {
                    "School_Type_Public": student_row["School_Type_Public"],
                    "learning_disabilities": student_row["Learning_Disabilities"]
                }
                
                payload = build_genai_payload(pd.Series(input_row))
                report = generate_teacher_explanation(payload)
                
                # Display AI report in a clean card
                with st.container():
                    st.markdown(
                        f'<div style="padding: 20px; background: rgba(0, 20, 51, 0.5); border-radius: 12px; border: 1px solid rgba(0, 102, 255, 0.3); margin-top: 20px;">'
                        '<div style="display: flex; align-items: center; margin-bottom: 15px;">'
                        '<div style="width: 40px; height: 40px; background: linear-gradient(135deg, #0066FF, #3B82F6); border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-right: 15px;">'
                        '<span style="font-size: 20px;">🤖</span>'
                        '</div>'
                        '<div>'
                        '<div style="font-size: 18px; font-weight: bold; color: #E5F0FF;">Personalized Intervention Plan</div>'
                        '<div style="font-size: 12px; color: #8CA3C7;">Generated by PARIX AI</div>'
                        '</div>'
                        '</div>'
                        '<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(0, 102, 255, 0.3), transparent); margin: 15px 0;"></div>'
                        f'<div style="color: #E5F0FF; line-height: 1.6;">{report.replace(chr(10), "<br>")}</div>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                
                # Add reset button
                #if st.button("🔄 Generate Another"):
                 #   st.session_state.generate_playbook = False
                  #  st.rerun()
                    
            except Exception as e:
                st.error(f"⚠️ Error generating insights: {e}")
                st.session_state.generate_playbook = False
elif screen == "📈 Class Insights":
    st.markdown("""
    <h1>📈 Class Insights & Analytics</h1>
    <p style="color: var(--text-muted);">Aggregated views and systemic patterns across the entire class</p>
    """, unsafe_allow_html=True)
    
    # Class-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👥 Total Students</div>
            <div class="metric-value">{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 Avg. Attendance</div>
            <div class="metric-value">{df['Attendance'].mean():.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Avg. Motivation</div>
            <div class="metric-value">{df['Motivation_Level'].mean():.1f}/2</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎯 Dominant Persona</div>
            <div style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0; color: var(--primary);">
                {df['failure_mode_persona'].mode()[0] if not df['failure_mode_persona'].empty else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Persona Distribution
    st.markdown("### 👥 Persona Distribution")
    persona_counts = df['failure_mode_persona'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar( # pyright: ignore[reportUndefinedVariable]
            x=persona_counts.index,
            y=persona_counts.values,
            color=persona_counts.index,
            color_discrete_sequence=['#8B5CF6', '#10B981', '#3B82F6', '#F59E0B'],
            labels={'x': 'Persona', 'y': 'Count'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E5F0FF'),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>📋 Persona Insights</h4>
            <p style="font-size: 0.9rem; color: var(--text-muted);">
            Understanding the dominant personas in your class helps tailor group interventions and classroom strategies.
            </p>
            <div class="custom-divider"></div>
            <h5>Recommendations:</h5>
            <ul style="font-size: 0.85rem; color: var(--text-light);">
                <li>Group students by persona for targeted workshops</li>
                <li>Design persona-specific resources</li>
                <li>Schedule persona-based parent meetings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Resource Analysis
    st.markdown("### 💡 Resource Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        internet_access = df['Internet_Access'].value_counts(normalize=True) * 100
        fig = px.pie( # pyright: ignore[reportUndefinedVariable]
            values=internet_access.values,
            names=internet_access.index,
            color=internet_access.index,
            color_discrete_map={'Yes': '#10B981', 'No': '#EF553B'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E5F0FF'),
            title="Internet Access Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        resource_mismatch = df['resource_mismatch_flag'].value_counts()
        fig = px.bar( # pyright: ignore[reportUndefinedVariable]
            x=resource_mismatch.index,
            y=resource_mismatch.values,
            color=resource_mismatch.index,
            color_discrete_map={'HIGH': '#EF553B', 'MEDIUM': '#FECB52', 'LOW': '#00CC96'},
            labels={'x': 'Resource Mismatch Level', 'y': 'Count'}
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E5F0FF'),
            title="Resource Mismatch Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download full report
    st.markdown("### 📥 Export Class Report")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("""
        **Full Class Analytics Report** includes:
        - Complete student dataset with all calculated metrics
        - Persona distribution analysis
        - Resource mismatch analysis
        - Intervention recommendations summary
        """)
    with col2:
        if st.button("📊 Generate Full Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv,
                    file_name=f"parix_class_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )