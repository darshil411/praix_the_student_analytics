import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

def plot_risk_distribution(df):
    """Enhanced donut chart for class risk levels with better styling."""
    # Create risk categories
    df = df.copy()
    df['Risk_Level'] = df['effort_outcome_gap_z'].apply(
        lambda x: 'High Risk' if x <= -0.9 else ('Medium Risk' if x <= -0.5 else 'Low Risk')
    )
    
    # Get counts and percentages
    counts = df['Risk_Level'].value_counts().reset_index()
    counts['percentage'] = (counts['count'] / counts['count'].sum() * 100).round(1)
    
    # Color map with our design system colors
    color_map = {
        'High Risk': '#FF4B4B',  # Red
        'Medium Risk': '#F59E0B', # Warning orange
        'Low Risk': '#10B981'     # Green
    }
    
    # Create donut chart with better styling
    fig = px.pie(
        counts, 
        values='count', 
        names='Risk_Level',
        hole=0.6,
        color='Risk_Level',
        color_discrete_map=color_map,
        labels={'count': 'Students', 'Risk_Level': 'Risk Level'}
    )
    
    # Add text annotations for better readability
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>%{value} students (%{percent})<extra></extra>',
        marker=dict(line=dict(color='#001433', width=2))
    )
    
    # Update layout for dark theme
    fig.update_layout(
        showlegend=True,
        margin=dict(t=20, b=20, l=20, r=20),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E5F0FF', family='Inter, sans-serif'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color='#8CA3C7')
        ),
        annotations=[
            dict(
                text=f"<b>Total</b><br>{len(df)}",
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False,
                font_color='#E5F0FF'
            )
        ]
    )
    
    return fig

def plot_priority_scatter(df):
    """
   
    This chart plots student personas against risk categories, with bubble size representing the number of students and color indicating the average potential score improvement. It provides a clear visual representation of where to focus intervention efforts based on both risk and expected impact.
    """
    df = df.copy()
    
    # Create risk categories for better grouping
    df['Risk_Category'] = pd.cut(
        df['effort_outcome_gap_z'],
        bins=[-float('inf'), -0.9, -0.5, float('inf')],
        labels=['High', 'Medium', 'Low']
    )
    
    # Map resource mismatch to numeric for grouping
    df['Resource_Level'] = df['resource_mismatch_flag'].map({'LOW': 1, 'MEDIUM': 2, 'HIGH': 3})
    
    # Group by persona and risk for better insights
    grouped = df.groupby(['failure_mode_persona', 'Risk_Category']).agg({
        'Student_ID': 'count',
        'effort_outcome_gap_z': 'mean',
        'expected_score_improvement': 'mean'
    }).reset_index()
    
    # Create bubble heatmap
    fig = go.Figure()
    
    # Size mapping for bubbles
    max_count = grouped['Student_ID'].max()
    min_size, max_size = 20, 80
    grouped['size'] = (grouped['Student_ID'] / max_count * (max_size - min_size)) + min_size
    
    # Color scale based on expected improvement
    colors = grouped['expected_score_improvement']
    
    # Add bubble traces for each persona
    personas = grouped['failure_mode_persona'].unique()
    for persona in personas:
        persona_data = grouped[grouped['failure_mode_persona'] == persona]
        
        fig.add_trace(go.Scatter(
            x=persona_data['Risk_Category'],
            y=persona_data['failure_mode_persona'],
            mode='markers+text',
            marker=dict(
                size=persona_data['size'],
                color=persona_data['expected_score_improvement'],
                colorscale='RdYlGn',  # Red-Yellow-Green scale
                showscale=True,
                colorbar=dict(
                    title="Avg. Potential<br>Improvement",
                    titlefont=dict(color='#8CA3C7', size=10),
                    tickfont=dict(color='#8CA3C7', size=9)
                ),
                line=dict(width=2, color='#001433')
            ),
            text=persona_data['Student_ID'].astype(str),
            textposition="middle center",
            textfont=dict(color='white', size=10, weight='bold'),
            name=persona,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Risk: %{x}<br>"
                "Students: %{text}<br>"
                "Avg. Gap: %{customdata[0]:.2f}<br>"
                "Avg. Potential: +%{customdata[1]:.1f}"
                "<extra></extra>"
            ),
            customdata=persona_data[['effort_outcome_gap_z', 'expected_score_improvement']].values
        ))
    
    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text="<b>📊 Priority Intervention Matrix</b><br><span style='font-size:12px;color:#8CA3C7'>Bubble size = # of students, Color = Potential improvement</span>",
            font=dict(color='#E5F0FF', size=18),
            x=0.5,
            y=0.95
        ),
        xaxis=dict(
            title="Risk Category",
            titlefont=dict(color='#8CA3C7', size=12),
            tickfont=dict(color='#8CA3C7', size=11),
            gridcolor='rgba(140, 163, 199, 0.1)',
            showgrid=True,
            categoryorder='array',
            categoryarray=['High', 'Medium', 'Low']
        ),
        yaxis=dict(
            title="Student Persona",
            titlefont=dict(color='#8CA3C7', size=12),
            tickfont=dict(color='#8CA3C7', size=11),
            gridcolor='rgba(140, 163, 199, 0.1)',
            showgrid=True
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(t=100, b=50, l=100, r=50),
        hovermode='closest',
        showlegend=False,
        font=dict(family='Inter, sans-serif')
    )
    
    # Add risk level annotations
    '''risk_colors = {'High': '#FF4B4B', 'Medium': '#F59E0B', 'Low': '#10B981'}
    for idx, risk in enumerate(['High', 'Medium', 'Low']):
        fig.add_annotation(
            x=risk,
            y=-0.9,  # Position below the x-axis
            text=f"<b>{risk}</b>",
            showarrow=False,
            font=dict(color=risk_colors[risk], size=11),
            xref="x",
            yref="paper"
        )'''
    
    return fig

def plot_student_radar(student_row, df_avg):
    """Enhanced radar chart with better styling and more metrics."""
    # Define comprehensive metrics for comparison
    categories = ['Attendance', 'Sleep Quality', 'Motivation', 'Resources', 'Study Hours', 'Previous Scores']
    
    # Normalize values for better visualization (0-100 scale)
    def normalize(value, min_val, max_val):
        return ((value - min_val) / (max_val - min_val)) * 100 if max_val > min_val else 50
    
    # Get min and max for normalization
    attendance_range = (df_avg['Attendance'].min(), df_avg['Attendance'].max())
    sleep_range = (df_avg['Sleep_Hours'].min(), df_avg['Sleep_Hours'].max())
    motivation_range = (df_avg['Motivation_Level'].min(), df_avg['Motivation_Level'].max())
    resources_range = (df_avg['resource_index'].min(), df_avg['resource_index'].max())
    study_range = (df_avg['Hours_Studied'].min(), df_avg['Hours_Studied'].max())
    scores_range = (df_avg['Previous_Scores'].min(), df_avg['Previous_Scores'].max())
    
    # Calculate student values
    student_vals = [
        normalize(student_row['Attendance'], *attendance_range),
        normalize(student_row['Sleep_Hours'] * 1.5, sleep_range[0] * 1.5, sleep_range[1] * 1.5),
        normalize(student_row['Motivation_Level'] * 10, motivation_range[0] * 10, motivation_range[1] * 10),
        normalize(student_row['resource_index'] * 100, resources_range[0] * 100, resources_range[1] * 100),
        normalize(student_row['Hours_Studied'], *study_range),
        normalize(student_row['Previous_Scores'], *scores_range)
    ]
    
    # Calculate class average values
    avg_vals = [
        normalize(df_avg['Attendance'].mean(), *attendance_range),
        normalize(df_avg['Sleep_Hours'].mean() * 1.5, sleep_range[0] * 1.5, sleep_range[1] * 1.5),
        normalize(df_avg['Motivation_Level'].mean() * 10, motivation_range[0] * 10, motivation_range[1] * 10),
        normalize(df_avg['resource_index'].mean() * 100, resources_range[0] * 100, resources_range[1] * 100),
        normalize(df_avg['Hours_Studied'].mean(), *study_range),
        normalize(df_avg['Previous_Scores'].mean(), *scores_range)
    ]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add student trace
    fig.add_trace(go.Scatterpolar(
        r=student_vals,
        theta=categories,
        fill='toself',
        name='This Student',
        fillcolor='rgba(0, 102, 255, 0.3)',
        line=dict(color='#0066FF', width=3),
        marker=dict(size=8, color='#0066FF')
    ))
    
    # Add class average trace
    fig.add_trace(go.Scatterpolar(
        r=avg_vals,
        theta=categories,
        fill='toself',
        name='Class Average',
        fillcolor='rgba(140, 163, 199, 0.2)',
        line=dict(color='#8CA3C7', width=2, dash='dash'),
        marker=dict(size=6, color='#8CA3C7')
    ))
    
    # Update layout with dark theme styling
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(140, 163, 199, 0.2)',
                linecolor='rgba(140, 163, 199, 0.3)',
                tickfont=dict(color='#8CA3C7', size=10),
                tickmode='array',
                tickvals=[0, 25, 50, 75, 100],
                ticktext=['0%', '25%', '50%', '75%', '100%']
            ),
            angularaxis=dict(
                gridcolor='rgba(140, 163, 199, 0.2)',
                linecolor='rgba(140, 163, 199, 0.3)',
                tickfont=dict(color='#E5F0FF', size=11)
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color='#8CA3C7', size=12),
            bgcolor='rgba(0, 20, 51, 0.7)',
            bordercolor='rgba(0, 102, 255, 0.3)',
            borderwidth=1
        ),
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#E5F0FF')
    )
    
    return fig

def plot_intervention_impact(df):
    """New: Bar chart showing potential impact of different interventions."""
    if 'primary_lever' not in df.columns:
        return go.Figure()
    
    # Group by primary lever to show potential impact
    lever_impact = df.groupby('primary_lever').agg({
        'expected_score_improvement': 'mean',
        'Student_ID': 'count'
    }).sort_values('expected_score_improvement', ascending=True).reset_index()
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=lever_impact['primary_lever'],
        x=lever_impact['expected_score_improvement'],
        orientation='h',
        marker=dict(
            color=lever_impact['expected_score_improvement'],
            colorscale='RdYlGn',
            line=dict(color='#001433', width=1)
        ),
        text=lever_impact.apply(lambda x: f"+{x['expected_score_improvement']:.1f} ({x['Student_ID']} students)", axis=1),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Avg. Improvement: +%{x:.1f}<br>Affected Students: %{customdata}<extra></extra>",
        customdata=lever_impact['Student_ID']
    ))
    
    fig.update_layout(
        title=dict(
            text="<b>🎯 Intervention Impact Analysis</b>",
            font=dict(color='#E5F0FF', size=16),
            x=0.5
        ),
        xaxis=dict(
            title="Average Potential Score Improvement",
            titlefont=dict(color='#8CA3C7', size=12),
            tickfont=dict(color='#8CA3C7', size=11),
            gridcolor='rgba(140, 163, 199, 0.1)',
            showgrid=True
        ),
        yaxis=dict(
            title="Primary Intervention Lever",
            titlefont=dict(color='#8CA3C7', size=12),
            tickfont=dict(color='#8CA3C7', size=11)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(t=60, b=50, l=150, r=50),
        font=dict(family='Inter, sans-serif')
    )
    
    return fig

def plot_persona_breakdown(df):
    """New: Detailed breakdown of student personas."""
    if 'failure_mode_persona' not in df.columns:
        return go.Figure()
    
    # Get persona distribution with key metrics
    persona_stats = df.groupby('failure_mode_persona').agg({
        'Student_ID': 'count',
        'effort_outcome_gap_z': 'mean',
        'expected_score_improvement': 'mean',
        'Attendance': 'mean',
        'Motivation_Level': 'mean'
    }).reset_index()
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>Distribution</b>", "<b>Key Metrics</b>"),
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        horizontal_spacing=0.15
    )
    
    # Pie chart for distribution
    fig.add_trace(
        go.Pie(
            labels=persona_stats['failure_mode_persona'],
            values=persona_stats['Student_ID'],
            hole=0.4,
            marker=dict(colors=['#8B5CF6', '#10B981', '#3B82F6', '#F59E0B']),
            hovertemplate="<b>%{label}</b><br>%{value} students (%{percent})<extra></extra>",
            textinfo='percent+label',
            textposition='inside'
        ),
        row=1, col=1
    )
    
    # Bar chart for average improvement
    fig.add_trace(
        go.Bar(
            x=persona_stats['failure_mode_persona'],
            y=persona_stats['expected_score_improvement'],
            marker=dict(
                color=persona_stats['expected_score_improvement'],
                colorscale='RdYlGn',
                line=dict(color='#001433', width=1)
            ),
            text=[f"+{val:.1f}" for val in persona_stats['expected_score_improvement']],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>Avg. Potential: +%{y:.1f}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>👥 Student Persona Analysis</b>",
            font=dict(color='#E5F0FF', size=18),
            x=0.5,
            y=0.95
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        margin=dict(t=100, b=50, l=50, r=50),
        showlegend=False,
        font=dict(family='Inter, sans-serif', color='#E5F0FF')
    )
    
    # Update subplot titles
    fig.update_annotations(
        font=dict(color='#8CA3C7', size=14)
    )
    
    # Update axes
    fig.update_xaxes(
        row=1, col=2,
        tickfont=dict(color='#8CA3C7', size=11)
    )
    fig.update_yaxes(
        row=1, col=2,
        title="Avg. Potential Improvement",
        titlefont=dict(color='#8CA3C7', size=12),
        tickfont=dict(color='#8CA3C7', size=11),
        gridcolor='rgba(140, 163, 199, 0.1)'
    )
    
    return fig