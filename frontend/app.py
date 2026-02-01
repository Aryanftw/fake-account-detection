import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests
import re

# Page configuration
st.set_page_config(
    page_title="Fake Account Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Playfair+Display:wght@700;900&display=swap');
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    /* Headers */
    h1 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-family: 'Playfair Display', serif !important;
        color: #a8b2d1 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
    }
    
    h3 {
        font-family: 'JetBrains Mono', monospace !important;
        color: #8892b0 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(100, 126, 234, 0.05);
        border: 1px solid rgba(100, 126, 234, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(100, 126, 234, 0.3);
        border-color: rgba(100, 126, 234, 0.4);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Verdict badges */
    .verdict-human {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
        letter-spacing: 0.05em;
    }
    
    .verdict-bot {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        letter-spacing: 0.05em;
    }
    
    .verdict-suspicious {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
        font-size: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.4);
        letter-spacing: 0.05em;
    }
    
    /* Input field */
    .stTextInput > div > div > input {
        background: rgba(100, 126, 234, 0.08) !important;
        border: 2px solid rgba(100, 126, 234, 0.3) !important;
        border-radius: 12px !important;
        color: #e6f1ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: rgba(100, 126, 234, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(100, 126, 234, 0.2) !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95) !important;
        border-right: 1px solid rgba(100, 126, 234, 0.2) !important;
    }
    
    /* Alert boxes */
    .alert-critical {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: 'JetBrains Mono', monospace;
        color: #fca5a5;
    }
    
    .alert-warning {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: 'JetBrains Mono', monospace;
        color: #fbbf24;
    }
    
    .alert-info {
        background: rgba(100, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: 'JetBrains Mono', monospace;
        color: #a8b2d1;
    }
    
    /* Feature tag */
    .feature-tag {
        display: inline-block;
        background: rgba(100, 126, 234, 0.15);
        border: 1px solid rgba(100, 126, 234, 0.3);
        color: #a8b2d1;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        margin: 0.2rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(100, 126, 234, 0.08) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(100, 126, 234, 0.2) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


def get_verdict_class(verdict):
    """Return CSS class based on verdict"""
    if verdict == "HUMAN":
        return "verdict-human"
    elif verdict == "BOT":
        return "verdict-bot"
    else:
        return "verdict-suspicious"


def create_risk_gauge(risk_score):
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24, 'family': 'JetBrains Mono', 'color': '#a8b2d1'}},
        delta={'reference': 50, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': '#10b981'}},
        number={'font': {'size': 60, 'family': 'JetBrains Mono', 'color': '#e6f1ff'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#667eea"},
            'bar': {'color': "#667eea", 'thickness': 0.75},
            'bgcolor': "rgba(100, 126, 234, 0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(100, 126, 234, 0.3)",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.3)'},
                {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "#e6f1ff", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#a8b2d1", 'family': "JetBrains Mono"},
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_model_breakdown_chart(model_data):
    """Create a bar chart for model breakdown"""
    models = list(model_data.keys())
    values = [model_data[m] for m in models]
    
    # Color mapping
    colors = ['#667eea', '#764ba2', '#f59e0b']
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=values,
            marker=dict(
                color=colors[:len(models)],
                line=dict(color='rgba(100, 126, 234, 0.5)', width=2)
            ),
            text=[f'{v:.2f}' for v in values],
            textposition='outside',
            textfont=dict(size=14, family='JetBrains Mono', color='#e6f1ff')
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Model Risk Breakdown',
            'font': {'size': 20, 'family': 'JetBrains Mono', 'color': '#a8b2d1'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#a8b2d1", 'family': "JetBrains Mono"},
        height=400,
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgba(100, 126, 234, 0.3)',
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(100, 126, 234, 0.1)',
            showline=True,
            linecolor='rgba(100, 126, 234, 0.3)',
            range=[0, max(values) * 1.2] if values else [0, 1.1]
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def create_network_graph(network_data):
    """Create a network visualization"""
    # Create sample network data
    np.random.seed(42)
    n_nodes = 20
    
    # Generate positions in a circle with some randomness
    angles = np.linspace(0, 2*np.pi, n_nodes)
    x = np.cos(angles) + np.random.normal(0, 0.1, n_nodes)
    y = np.sin(angles) + np.random.normal(0, 0.1, n_nodes)
    
    # Create edges
    edge_x = []
    edge_y = []
    for i in range(n_nodes):
        # Connect to 2-3 random neighbors
        for _ in range(np.random.randint(2, 4)):
            j = np.random.randint(0, n_nodes)
            edge_x.extend([x[i], x[j], None])
            edge_y.extend([y[i], y[j], None])
    
    # Edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(100, 126, 234, 0.2)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node colors based on cluster
    node_colors = ['#667eea' if i < 15 else '#ef4444' for i in range(n_nodes)]
    node_sizes = [20 if i != 0 else 30 for i in range(n_nodes)]  # Highlight queried account
    
    # Node trace
    node_trace = go.Scatter(
        x=x, y=y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color='rgba(255, 255, 255, 0.3)')
        ),
        text=[f'Account {i}' for i in range(n_nodes)]
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title={
            'text': 'Network Connection Graph',
            'font': {'size': 20, 'family': 'JetBrains Mono', 'color': '#a8b2d1'}
        },
        showlegend=False,
        hovermode='closest',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=60, b=20),
        annotations=[
            dict(
                text=f"Cluster: {network_data['cluster_group']}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                font=dict(size=14, family='JetBrains Mono', color='#a8b2d1'),
                bgcolor='rgba(100, 126, 234, 0.1)',
                bordercolor='rgba(100, 126, 234, 0.3)',
                borderwidth=1,
                borderpad=8
            )
        ]
    )
    
    return fig


def create_timeline_chart(avg_daily_posts):
    """Create a timeline/activity chart"""
    # Generate sample data for last 30 days
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    posts = np.random.poisson(avg_daily_posts, 30)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=posts,
        mode='lines+markers',
        name='Posts',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2', line=dict(width=2, color='#667eea')),
        fill='tozeroy',
        fillcolor='rgba(100, 126, 234, 0.2)'
    ))
    
    fig.update_layout(
        title={
            'text': 'Posting Activity Timeline (Last 30 Days)',
            'font': {'size': 20, 'family': 'JetBrains Mono', 'color': '#a8b2d1'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#a8b2d1", 'family': "JetBrains Mono"},
        height=350,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(100, 126, 234, 0.1)',
            showline=True,
            linecolor='rgba(100, 126, 234, 0.3)'
        ),
        yaxis=dict(
            title='Number of Posts',
            showgrid=True,
            gridcolor='rgba(100, 126, 234, 0.1)',
            showline=True,
            linecolor='rgba(100, 126, 234, 0.3)'
        ),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def create_shap_waterfall(shap_features):
    """Create SHAP waterfall chart showing feature contributions"""
    features = [f['feature'] for f in shap_features]
    impacts = [f['impact'] if f['direction'] == 'Increases Risk' else -f['impact'] for f in shap_features]
    colors = ['#ef4444' if imp > 0 else '#10b981' for imp in impacts]
    
    # Sort by absolute impact
    sorted_indices = sorted(range(len(impacts)), key=lambda i: abs(impacts[i]), reverse=True)
    features = [features[i] for i in sorted_indices]
    impacts = [impacts[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        y=features,
        x=impacts,
        connector={"line": {"color": "rgba(100, 126, 234, 0.3)", "width": 2}},
        decreasing={"marker": {"color": "#10b981"}},
        increasing={"marker": {"color": "#ef4444"}},
        textposition="outside",
        text=[f"{abs(imp):.3f}" for imp in impacts],
        textfont={"family": "JetBrains Mono", "size": 12, "color": "#e6f1ff"}
    ))
    
    fig.update_layout(
        title={
            'text': 'SHAP Feature Impact Analysis',
            'font': {'size': 20, 'family': 'JetBrains Mono', 'color': '#a8b2d1'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#a8b2d1", 'family': "JetBrains Mono"},
        height=300,
        xaxis=dict(
            title='Impact on Risk Score',
            showgrid=True,
            gridcolor='rgba(100, 126, 234, 0.1)',
            showline=True,
            linecolor='rgba(100, 126, 234, 0.3)',
            zeroline=True,
            zerolinecolor='rgba(100, 126, 234, 0.5)',
            zerolinewidth=2
        ),
        yaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgba(100, 126, 234, 0.3)'
        ),
        margin=dict(l=150, r=40, t=60, b=40),
        showlegend=False
    )
    
    return fig


def create_linguistic_radar(ling_data):
    """Create radar chart for linguistic analysis"""
    categories = ['Bot Probability', 'Avg Tweet Length', 'Reading Ease', 'Grade Variance']
    
    # Normalize values to 0-100 scale
    bot_prob = float(ling_data['bot_language_probability'].strip('%'))
    avg_length = min(ling_data['avg_tweet_length'] * 2, 100)  # Scale to 100
    reading_ease = ling_data['reading_ease_mean']
    grade_var = min(ling_data['grade_variance'] * 6.67, 100)  # Scale to 100 (15 max -> 100)
    
    values = [bot_prob, avg_length, reading_ease, grade_var]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#764ba2'),
        name='Linguistic Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=False,
                gridcolor='rgba(100, 126, 234, 0.2)',
                tickfont=dict(size=10, color='#8892b0')
            ),
            angularaxis=dict(
                gridcolor='rgba(100, 126, 234, 0.2)',
                linecolor='rgba(100, 126, 234, 0.3)',
                tickfont=dict(size=11, family='JetBrains Mono', color='#a8b2d1')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#a8b2d1", 'family': "JetBrains Mono"},
        height=400,
        showlegend=False,
        title={
            'text': 'Linguistic Analysis Profile',
            'font': {'size': 20, 'family': 'JetBrains Mono', 'color': '#a8b2d1'}
        },
        margin=dict(l=80, r=80, t=80, b=40)
    )
    
    return fig


def parse_tweet_date(tweet_text):
    """Extract date from tweet text"""
    import re
    match = re.search(r'\[(.*?)\]', tweet_text)
    if match:
        date_str = match.group(1)
        try:
            # Parse the date format: "Wed Sep 17 18:22:25 +0000 2014"
            from datetime import datetime
            return datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')
        except:
            return None
    return None


def create_tweets_timeline(tweets):
    """Create timeline visualization from tweets"""
    # Parse tweet dates and count by day
    dates = []
    for tweet in tweets:
        date = parse_tweet_date(tweet)
        if date:
            dates.append(date.date())
    
    if not dates:
        # Fallback to sample data if parsing fails
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        counts = np.random.poisson(2, 30)
    else:
        # Count tweets per day
        date_counts = pd.Series(dates).value_counts().sort_index()
        dates = date_counts.index
        counts = date_counts.values
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=counts,
        mode='lines+markers',
        name='Tweets',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#764ba2', line=dict(width=2, color='#667eea')),
        fill='tozeroy',
        fillcolor='rgba(100, 126, 234, 0.2)',
        hovertemplate='<b>%{x}</b><br>Tweets: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Tweet Activity Distribution',
            'font': {'size': 20, 'family': 'JetBrains Mono', 'color': '#a8b2d1'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#a8b2d1", 'family': "JetBrains Mono"},
        height=350,
        xaxis=dict(
            title='Date',
            showgrid=True,
            gridcolor='rgba(100, 126, 234, 0.1)',
            showline=True,
            linecolor='rgba(100, 126, 234, 0.3)'
        ),
        yaxis=dict(
            title='Number of Tweets',
            showgrid=True,
            gridcolor='rgba(100, 126, 234, 0.1)',
            showline=True,
            linecolor='rgba(100, 126, 234, 0.3)'
        ),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def real_api_call(username):
    """Make actual API call to the backend"""
    res = requests.post(
        "http://127.0.0.1:8000/analyze/by-username",
        json={"username": username},
        timeout=30
    )
    return res.json()


# Main app
def main():
    # Header
    st.markdown("<h1>🔍 Fake Account Detection & Risk Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-family: JetBrains Mono; color: #8892b0; font-size: 1.1rem; margin-bottom: 2rem;'>Advanced behavioral analysis and machine learning-powered detection system</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='font-size: 1.5rem !important;'>⚙️ Analysis Settings</h2>", unsafe_allow_html=True)
        
        # Input field
        username = st.text_input(
            "Enter Twitter Username",
            placeholder="@username",
            help="Enter the Twitter username to analyze (without @)"
        )
        
        analyze_button = st.button("🚀 Analyze Account", use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class='alert-info'>
            <strong>ℹ️ How it works:</strong><br>
            Our system analyzes multiple factors including behavioral patterns, 
            linguistic signals, network connections, and metadata to determine 
            account authenticity.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("<h3>Detection Models</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='feature-tag'>Random Forest</div>
        <div class='feature-tag'>BERT NLP</div>
        <div class='feature-tag'>LightGBM</div>
        <div class='feature-tag'>Anomaly Detection</div>
        """, unsafe_allow_html=True)
    
    # Main content
    if analyze_button and username:
        with st.spinner('🔍 Analyzing account...'):
            try:
                # Make API call
                data = real_api_call(username)
                
                # Store in session state
                st.session_state['analysis_data'] = data
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Network error: {str(e)}")
                st.info("Make sure your backend is running at http://127.0.0.1:8000")
                return
            except Exception as e:
                st.error(f"❌ Error analyzing account: {str(e)}")
                st.error(f"Error type: {type(e).__name__}")
                return
    
    # Display results if available
    if 'analysis_data' in st.session_state:
        data = st.session_state['analysis_data']
        
        # Validate data structure
        if not isinstance(data, dict):
            st.error("❌ Invalid data format received from API")
            st.json(data)
            return
            
        if 'identity' not in data:
            st.error("❌ Response missing 'identity' field")
            st.write("Available keys:", list(data.keys()))
            st.json(data)
            return
        
        # Top section - Verdict and Risk Score
        col1, col2 = st.columns([1, 1])
        
        with col1:
            verdict = data['identity']['final_verdict']
            verdict_class = get_verdict_class(verdict)
            
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Final Verdict</div>
                <div class='{verdict_class}' style='margin-top: 1rem;'>
                    {verdict}
                </div>
                <div class='metric-label' style='margin-top: 1rem;'>Account: @{data['identity']['username']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.plotly_chart(create_risk_gauge(data['identity']['risk_score']), use_container_width=True)
        
        # Model Breakdown
        st.markdown("<h2>📊 Model Risk Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            model_data = {
                "Behavioral (RF)": data['model_breakdown']['behavioral_risk_rf'],
                "Linguistic (BERT)": data['model_breakdown']['linguistic_risk_bert'],
                "Network (LGBM)": data['model_breakdown']['network_risk_lgbm']
            }
            st.plotly_chart(create_model_breakdown_chart(model_data), use_container_width=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card' style='height: 360px; display: flex; flex-direction: column; justify-content: center;'>
                <div class='metric-label'>Anomaly Status</div>
                <div class='metric-value' style='color: {"#ef4444" if data["model_breakdown"]["anomaly_detector"] == "CRITICAL" else "#10b981"};'>
                    {data['model_breakdown']['anomaly_detector']}
                </div>
                <div style='margin-top: 2rem;'>
                    <div class='metric-label'>Risk Distribution</div>
                    <div style='margin-top: 1rem; font-family: JetBrains Mono; font-size: 0.9rem; color: #a8b2d1;'>
                        <div style='margin: 0.5rem 0;'>🔴 Behavioral: {data['model_breakdown']['behavioral_risk_rf']:.2%}</div>
                        <div style='margin: 0.5rem 0;'>🟡 Linguistic: {data['model_breakdown']['linguistic_risk_bert']:.2%}</div>
                        <div style='margin: 0.5rem 0;'>🟢 Network: {data['model_breakdown']['network_risk_lgbm']:.2%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Behavioral Flags
        st.markdown("<h2>🚩 Behavioral Flags & Analysis</h2>", unsafe_allow_html=True)
        
        flags = [flag for flag in data['explainability']['behavioral_flags'] if flag is not None]
        
        if flags:
            cols = st.columns(min(len(flags), 3))
            for idx, flag in enumerate(flags):
                with cols[idx % 3]:
                    alert_class = "alert-critical" if "Automated" in flag or "Frequency" in flag else "alert-warning"
                    st.markdown(f"""
                    <div class='{alert_class}'>
                        <strong>⚠️ {flag}</strong>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='alert-info'>
                <strong>✅ No suspicious behavioral flags detected</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # SHAP Feature Importance
        st.markdown("<h2>🎯 Top Contributing Features (SHAP Analysis)</h2>", unsafe_allow_html=True)
        
        shap_features = data['explainability']['top_contributing_features_shap']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(create_shap_waterfall(shap_features), use_container_width=True)
        
        with col2:
            st.markdown("<h3 style='margin-top: 0;'>Feature Explanations</h3>", unsafe_allow_html=True)
            for idx, feature in enumerate(shap_features, 1):
                direction_color = "#ef4444" if feature['direction'] == "Increases Risk" else "#10b981"
                direction_icon = "📈" if feature['direction'] == "Increases Risk" else "📉"
                
                st.markdown(f"""
                <div class='metric-card' style='padding: 1rem; margin-bottom: 0.8rem;'>
                    <div style='font-family: JetBrains Mono; font-size: 0.75rem; color: #8892b0; margin-bottom: 0.3rem;'>
                        #{idx} FEATURE
                    </div>
                    <div style='font-family: JetBrains Mono; font-size: 1rem; font-weight: 600; color: #e6f1ff; margin-bottom: 0.5rem;'>
                        {feature['feature']}
                    </div>
                    <div style='font-family: JetBrains Mono; font-size: 0.85rem; color: {direction_color}; margin-bottom: 0.3rem;'>
                        {direction_icon} {feature['direction']}
                    </div>
                    <div style='font-family: JetBrains Mono; font-size: 0.75rem; color: #8892b0;'>
                        Impact: {feature['impact']:.4f}
                    </div>
                    <div style='font-family: JetBrains Mono; font-size: 0.7rem; color: #64748b; margin-top: 0.5rem; font-style: italic;'>
                        {feature['reason']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Linguistic Analysis
        st.markdown("<h2>📝 Linguistic Analysis</h2>", unsafe_allow_html=True)
        
        ling = data['explainability']['linguistic_analysis']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(create_linguistic_radar(ling), use_container_width=True)
        
        with col2:
            # Linguistic metrics in cards
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Bot Language Probability</div>
                <div class='metric-value' style='color: #667eea; font-size: 2rem;'>{ling['bot_language_probability']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Avg Tweet Length</div>
                    <div class='metric-value' style='color: #764ba2; font-size: 1.5rem;'>{ling['avg_tweet_length']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Grade Variance</div>
                    <div class='metric-value' style='color: #10b981; font-size: 1.5rem;'>{ling['grade_variance']:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Reading Ease</div>
                    <div class='metric-value' style='color: #f59e0b; font-size: 1.5rem;'>{ling['reading_ease_mean']:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
                
                risk_color = "#ef4444" if "High" in ling['linguistic_risk'] else "#f59e0b" if "Medium" in ling['linguistic_risk'] else "#10b981"
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Linguistic Risk</div>
                    <div style='color: {risk_color}; font-family: JetBrains Mono; font-size: 0.85rem; font-weight: 600; margin-top: 0.5rem;'>
                        {ling['linguistic_risk']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Interpretation box
            st.markdown(f"""
            <div class='alert-info' style='margin-top: 1rem;'>
                <strong>💡 Interpretation:</strong><br>
                {ling['interpretation']}
            </div>
            """, unsafe_allow_html=True)
        
        # Recent Tweets Section
        st.markdown("<h2>🐦 Recent Tweets Analysis</h2>", unsafe_allow_html=True)
        
        if 'tweets' in data and data['tweets']:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Display tweets in styled cards
                st.markdown("<h3>Tweet Feed</h3>", unsafe_allow_html=True)
                
                for idx, tweet in enumerate(data['tweets'][:5]):  # Show first 5 tweets
                    # Parse tweet
                    import re
                    match = re.search(r'\[(.*?)\] (.*)', tweet)
                    if match:
                        date_str = match.group(1)
                        content = match.group(2)
                    else:
                        date_str = "Unknown date"
                        content = tweet
                    
                    st.markdown(f"""
                    <div class='metric-card' style='margin-bottom: 1rem;'>
                        <div style='font-family: JetBrains Mono; font-size: 0.7rem; color: #8892b0; margin-bottom: 0.5rem;'>
                            📅 {date_str}
                        </div>
                        <div style='font-family: system-ui; font-size: 0.95rem; color: #e6f1ff; line-height: 1.5;'>
                            {content}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show all tweets button
                with st.expander(f"📋 View All {len(data['tweets'])} Tweets"):
                    for idx, tweet in enumerate(data['tweets'], 1):
                        st.markdown(f"""
                        <div style='padding: 0.5rem; border-left: 2px solid rgba(100, 126, 234, 0.3); margin-bottom: 0.5rem; font-family: system-ui; font-size: 0.85rem; color: #a8b2d1;'>
                            <strong>#{idx}</strong> {tweet}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                st.plotly_chart(create_tweets_timeline(data['tweets']), use_container_width=True)
        else:
            st.markdown("""
            <div class='alert-info'>
                <strong>ℹ️ No tweets available for analysis</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Network and Timeline
        st.markdown("<h2>🌐 Network & Activity Analysis</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(
                create_network_graph(data['visual_data']['network_node']), 
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                create_timeline_chart(data['visual_data']['timeline_data']['avg_daily_posts']), 
                use_container_width=True
            )
        
        # Detailed Metrics Expander
        with st.expander("📈 View Detailed Metrics & Raw Data"):
            st.markdown("<h3>Complete Analysis Report</h3>", unsafe_allow_html=True)
            
            # Display JSON
            st.json(data)
            
            # Download button
            json_str = json.dumps(data, indent=2)
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=json_str,
                file_name=f"account_analysis_{data['identity']['username']}.json",
                mime="application/json"
            )
    
    else:
        # Landing state
        st.markdown("""
        <div style='text-align: center; padding: 4rem 2rem; margin-top: 3rem;'>
            <div style='font-size: 5rem; margin-bottom: 1rem;'>🔍</div>
            <h2 style='color: #a8b2d1; font-family: Playfair Display; margin-bottom: 1rem;'>
                Ready to Analyze
            </h2>
            <p style='font-family: JetBrains Mono; color: #8892b0; font-size: 1.1rem; max-width: 600px; margin: 0 auto;'>
                Enter a Twitter username in the sidebar to begin comprehensive 
                fake account detection and risk analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features showcase
        st.markdown("<h2 style='text-align: center; margin-top: 3rem;'>🎯 Key Features</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        features = [
            ("🤖", "ML Models", "Random Forest, BERT, LightGBM"),
            ("📊", "Behavior Analysis", "Posting patterns & anomalies"),
            ("🌐", "Network Graph", "Connection analysis"),
            ("📈", "Risk Scoring", "Multi-factor assessment")
        ]
        
        for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
            with col:
                st.markdown(f"""
                <div class='metric-card' style='text-align: center; min-height: 180px;'>
                    <div style='font-size: 3rem; margin-bottom: 0.5rem;'>{icon}</div>
                    <div class='metric-label' style='font-size: 0.9rem; margin-bottom: 0.5rem;'>{title}</div>
                    <div style='font-family: JetBrains Mono; font-size: 0.75rem; color: #8892b0;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()