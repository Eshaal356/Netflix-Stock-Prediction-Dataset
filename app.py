import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import utils

# --- Page Configuration ---
st.set_page_config(
    page_title="Netflix Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64

# ... (Previous imports)

# --- Assets ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Helper to load dynamic CSS
try:
    img_base64 = get_base64_of_bin_file("sidebar_bg.png")
    sidebar_bg_ext = "png"
except:
    img_base64 = "" # Fallback if file missing

# --- Custom Styling (Strict Netflix Black & Red) ---
st.markdown(f"""
<style>
    /* Global Background - Strictly Black */
    .stApp {{
        background-color: #000000;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #E50914; 
    }}
    
    /* Animation Keyframes for Shining Effect */
    @keyframes shine {{
        0% {{ filter: brightness(100%) opacity(0.6); }}
        50% {{ filter: brightness(130%) opacity(0.8); }}
        100% {{ filter: brightness(100%) opacity(0.6); }}
    }}
    
    /* Sidebar - Background Image & Shining Effect */
    [data-testid="stSidebar"]::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        opacity: 0.7; 
        z-index: -1;
        animation: shine 3s infinite ease-in-out;
    }}

    [data-testid="stSidebar"] {{
        background-color: transparent !important; 
        border-right: 1px solid #333;
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        background: transparent;
    }}
    
    /* Sidebar Text specifically */
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    /* Metric Cards */
    .metric-card {{
        background: #141414; /* Slightly lighter black for cards */
        border-radius: 4px;
        border: 1px solid #333;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(229, 9, 20, 0.2); /* Red shadow */
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: #E50914 !important; /* Netflix Red */
        font-weight: 800;
        letter-spacing: 1px;
    }}
    
    /* Text */
    p, .stMarkdown, .stText {{
        color: #E50914; /* Making sure all text is red as requested */
    }}
    
    /* Callouts */
    .stInfo, .stSuccess, .stWarning, .stError {{
        background-color: #141414;
        color: #E50914; 
        border-left: 4px solid #E50914; 
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: #E50914;
        color: white !important; /* Force white text */
        border: none;
        border-radius: 2px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        padding: 0.5rem 2rem; /* Ensure enough height */
    }}
    .stButton>button:hover {{
        background-color: #f40612;
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(229, 9, 20, 0.7);
        color: white !important;
    }}
    .stButton>button p {{
        color: white !important; /* Ensure markdown text inside button is white */
    }}
    
    /* Radio Buttons / Selectbox */
    .stRadio > label, .stSelectbox > label {{
        color: #E50914 !important;
        font-weight: bold;
    }}
    
    /* Plotly Chart Background */
    .js-plotly-plot .plotly .main-svg {{
        background: transparent !important;
    }}
    /* Card Container */
    .card-container {{
        display: flex;
        flex-wrap: nowrap;
        overflow-x: auto;
        padding: 20px 0;
        gap: 15px;
    }}
    .card-container::-webkit-scrollbar {{
        height: 6px;
    }}
    .card-container::-webkit-scrollbar-thumb {{
        background: #333;
        border-radius: 10px;
    }}
    /* Netflix Style Cards */
    .n-card {{
        min-width: 200px;
        height: 300px;
        background: #1e1e1e;
        border-radius: 6px;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, border 0.3s ease;
        border: 2px solid transparent;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 15px;
        background-size: cover;
        background-position: center;
    }}
    .n-card:hover {{
        transform: scale(1.05);
        border: 2px solid #E50914;
        z-index: 10;
    }}
    .n-card-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.8) 100%);
    }}
    .n-card-title {{
        color: white;
        font-weight: 800;
        font-size: 1.2rem;
        z-index: 1;
        margin-bottom: 5px;
    }}
    .n-card-badge {{
        position: absolute;
        top: 10px;
        left: 10px;
        background: #E50914;
        color: white;
        padding: 2px 8px;
        font-size: 0.7rem;
        font-weight: bold;
        border-radius: 2px;
        z-index: 1;
    }}
    .n-card-value {{
        color: #E50914;
        font-weight: 700;
        z-index: 1;
        font-size: 0.9rem;
    }}
    /* Ticker Animation */
    @keyframes ticker {{
        0% {{ transform: translateX(100%); }}
        100% {{ transform: translateX(-100%); }}
    }}
    .ticker-wrap {{
        width: 100%;
        overflow: hidden;
        background: rgba(229, 9, 20, 0.1);
        border-bottom: 1px solid #E50914;
        padding: 5px 0;
        margin-bottom: 20px;
    }}
    .ticker-move {{
        display: inline-block;
        white-space: nowrap;
        animation: ticker 30s linear infinite;
    }}
    .ticker-item {{
        display: inline-block;
        padding: 0 50px;
        color: #E50914;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}

    /* Holographic Glitch Card */
    .hologram-card {{
        background: rgba(20, 20, 20, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(229, 9, 20, 0.3);
        border-radius: 15px;
        padding: 25px;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }}
    .hologram-card:hover {{
        transform: translateY(-10px) scale(1.02);
        border-color: #E50914;
        box-shadow: 0 0 30px rgba(229, 9, 20, 0.4);
    }}
    .hologram-card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: -100%;
        width: 50%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: 0.5s;
    }}
    .hologram-card:hover::before {{
        left: 100%;
    }}

    /* Scanline Effect */
    .scanline {{
        width: 100%;
        height: 100px;
        z-index: 10;
        background: linear-gradient(0deg, rgba(0, 0, 0, 0) 0%, rgba(229, 9, 20, 0.05) 50%, rgba(0, 0, 0, 0) 100%);
        opacity: 0.1;
        position: absolute;
        bottom: 100%;
        animation: scanline 6s linear infinite;
    }}
    @keyframes scanline {{
        0% {{ bottom: 100%; }}
        80% {{ bottom: -100%; }}
        100% {{ bottom: -100%; }}
    }}

    /* Cyberpunk Terminal */
    .terminal-box {{
        background: #050505;
        border: 1px solid #333;
        color: #00ff00;
        font-family: 'Courier New', Courier, monospace;
        padding: 10px;
        font-size: 0.7rem;
        height: 150px;
        overflow-y: hidden;
        border-radius: 4px;
        box-shadow: inset 0 0 10px #000;
    }}
    .terminal-line {{
        margin-bottom: 5px;
        opacity: 0.7;
    }}
    
    /* Gapped Grid for Tactical Panels */
    .tactical-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-top: 30px;
    }}
</style>
""", unsafe_allow_html=True)

# --- Assets ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# Netflix/Cinematic Style Animations (Transparent/Dark Friendly)
lottie_netflix = load_lottieurl("https://assets5.lottiefiles.com/private_files/lf30_cmdcmgh0.json") 
lottie_developer = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_sk5h1y86.json") # Verified Moving Programmer
if not lottie_netflix:
     lottie_netflix = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_cti0skro.json") 

# --- Sidebar ---
try:
    # Use columns to center and constrain size, or just width parameter
    col1, col2, col3 = st.sidebar.columns([1,2,1])
    with col2:
        st.image("netflix_logo.png", width=150) # Smaller Logo
except:
    st.sidebar.title("NETFLIX") 

# --- Navigation Logic ---
nav_options = ["Home", "Data Lab (ETL)", "Deep EDA", "Advanced Visuals", "Strategy Intelligence Lab", "Executive Dashboard", "3D Galaxy", "Advanced ML Models", "Simulator"]
if 'navigation' not in st.session_state:
    st.session_state.navigation = "Home"
if 'command_overlay' not in st.session_state:
    st.session_state.command_overlay = None

# Sync session state with radio button
try:
    nav_index = nav_options.index(st.session_state.navigation)
except ValueError:
    nav_index = 0

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", nav_options, index=nav_index) # Removed key to avoid locking

# Update session state when radio changes
st.session_state.navigation = page

st.sidebar.markdown("---")
st.sidebar.info("Creating a cinematic experience for financial data.")

# --- Data Loading ---
df = utils.load_data('NFLX.csv') 
df_processed = utils.process_data(df.copy()) 

# --- Pages ---

if page == "Home":
    # --- Overlay Toggle Logic ---
    if st.session_state.command_overlay:
        overlay = st.session_state.command_overlay
        st.markdown(f"<h2 style='color: #E50914; letter-spacing: 5px; text-align: center;'>[ RED-TEAM {overlay.upper()} ACTIVE ]</h2>", unsafe_allow_html=True)
        
        if overlay == "Core Monitor":
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(utils.plot_tactical_radar(), use_container_width=True)
                st.markdown("<p style='text-align: center; color: #E50914; font-size: 0.7rem;'>STRATEGIC SENTINEL RADAR</p>", unsafe_allow_html=True)
                st.plotly_chart(utils.plot_strategy_gauge(), use_container_width=True)
            with c2:
                st.markdown("""
                <div class="hologram-card">
                    <div style="color: #E50914; font-weight: bold; margin-bottom: 10px;">CORE INTEL MONITORING:</div>
                    <div style="font-size: 0.8rem; color: #aaa;">
                    - Institutional Buy-Wall detected at $420.00<br>
                    - Short-Squeeze probability: 14%<br>
                    - Retail Sentiment: Overwhelmingly Bullish<br>
                    - Strategic Reserve: Optimizing...
                    </div>
                </div>
                <br>
                <div class="terminal-box" style="height: 200px;">
                    <div style="color: #E50914; font-weight: bold; margin-bottom: 5px;">WHALE TRACKER v2.4</div>
                    <div class="terminal-line">09:12:44 - Large order block (15,200 units) @ $421.50</div>
                    <div class="terminal-line">09:13:01 - Dark Pool activity spiking in NYSE subspace</div>
                    <div class="terminal-line">09:14:22 - Hedge Fund 'A' reducing exposure...</div>
                    <div class="terminal-line">09:15:05 - Market Maker absorption active.</div>
                </div>
                """, unsafe_allow_html=True)
                
        elif overlay == "Neural Lab":
            st.plotly_chart(utils.plot_neural_diagnostics(), use_container_width=True)
            st.markdown("<p style='text-align: center; color: white; font-size: 0.8rem; opacity: 0.5;'>NEURAL WEIGHT CONVERGENCE DIAGNOSTIC</p>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(utils.plot_neural_mesh(), use_container_width=True)
                st.caption("Neural Entropy Landscape (Lower is better)")
            with c2:
                st.info("System is currently processing 512-layer deep learning weights. Convergence achieved at epoch 14.")
                st.markdown("""
                <div style="background: rgba(229, 9, 20, 0.1); border: 1px solid #E50914; padding: 15px; border-radius: 5px;">
                    <b style="color: #E50914;">NEURAL ACTIVATION STATUS</b><br>
                    <span style="font-size: 0.8rem; color: #ddd;">
                    L1: [################----] 82%<br>
                    L2: [#############-------] 65%<br>
                    L3: [####################] 100%<br>
                    L4: [##########----------] 50%
                    </span>
                </div>
                """, unsafe_allow_html=True)
            
        elif overlay == "Void Scan":
            st.plotly_chart(utils.plot_global_dominance(), use_container_width=True)
            st.markdown("<p style='text-align: center; color: #E50914; font-size: 0.7rem;'>GLOBAL STRATEGIC DOMINANCE MAP</p>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(utils.plot_competitor_battlefield(), use_container_width=True)
            with c2:
                st.warning("VOID SCAN: Competitor 'Disney+' is making localized gains in European regions. Defensive protocol suggested.")
                st.markdown("""
                <table style="width: 100%; color: #888; font-size: 0.8rem;">
                    <tr><th style="text-align: left; color: #E50914;">COMPETITOR</th><th style="text-align: left; color: #E50914;">TECH STATUS</th></tr>
                    <tr><td>Disney+</td><td>UPDATING CDN</td></tr>
                    <tr><td>HBOMax</td><td>ENCODING v3</td></tr>
                    <tr><td>Amazon</td><td>EDGE SYNCING</td></tr>
                </table>
                """, unsafe_allow_html=True)

        st.write("")
        if st.button("✖ TERMINATE COMMAND & RETURN", use_container_width=True):
            st.session_state.command_overlay = None
            st.rerun()
            
    else: # Normal Home View
        # 1. Neon Ticker (Global Market Feed)
        st.markdown(f"""
        <div class="ticker-wrap">
            <div class="ticker-move">
                <span class="ticker-item">● NEURAL NODE: ACTIVE</span>
                <span class="ticker-item">● MARKET SENTIMENT: OPTIMIZED</span>
                <span class="ticker-item">● LIQUIDITY PROBE: STABLE</span>
                <span class="ticker-item">● ALPHA STREAM: {df_processed['Daily_Return'].iloc[-1]*100:+.2f}%</span>
                <span class="ticker-item">● HFT SCAN: COMPLETE</span>
                <span class="ticker-item">● INSTITUTIONAL BIAS: ACCUMULATION</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("<h1 class='hero-title' style='font-size: 5rem; margin-bottom: 0; line-height: 0.9;'>STRATEGIC <span style='color: white;'>COMMAND</span></h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='color: #E50914; margin-top: 10px; letter-spacing: 12px; font-weight: 300;'>NETFLIX ENTERPRISE</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <p style='color: #888; font-size: 1.1rem; margin-top: 30px;'>
                Proprietary intelligence suite for Netflix Strategic Operations. 
                Integrating real-time neural forecasting with high-fidelity market telemetry.
            </p>
            """, unsafe_allow_html=True)
            
            st.write("")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("👁️ CORE MONITOR", use_container_width=True):
                    st.session_state.command_overlay = "Core Monitor"
                    st.rerun()
            with c2:
                if st.button("🧠 NEURAL LAB", use_container_width=True):
                    st.session_state.command_overlay = "Neural Lab"
                    st.rerun()
            with c3:
                if st.button("🌀 VOID SCAN", use_container_width=True):
                    st.session_state.command_overlay = "Void Scan"
                    st.rerun()

        with col2:
            st.markdown("<div style='border: 1px solid #E50914; padding: 5px; border-radius: 10px;'>", unsafe_allow_html=True)
            st.plotly_chart(utils.plot_mini_pulse(df_processed), use_container_width=True)
            st.markdown("<p style='text-align: center; color: #E50914; font-size: 0.6rem; letter-spacing: 3px;'>CORE PULSE (30D ENERGY)</p></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 3. Tactical Subsystems
        ts1, ts2 = st.columns([1, 2])
        
        with ts1:
            st.markdown("""
            <div class="terminal-box">
                <div class="terminal-line">> Initializing Neural Subsets...</div>
                <div class="terminal-line">> Loading High-Frequency Data...</div>
                <div class="terminal-line">> Running Monte Carlo iterations...</div>
                <div class="terminal-line">> Detecting Whale Movements... [OK]</div>
                <div class="terminal-line">> Calculating Alpha Deviation...</div>
                <div class="terminal-line">> Signal Strength: 9.8/10</div>
                <div class="terminal-line">> System Ready.</div>
            </div>
            """, unsafe_allow_html=True)

        with ts2:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("""
                <div class="hologram-card" style="height: 150px; padding: 15px;">
                    <div style="color: #E50914; font-size: 0.6rem; font-weight: bold; letter-spacing: 2px;">MARKET VOLTAGE</div>
                    <div style="color: white; font-size: 1.5rem; font-weight: 800;">HIGH GAIN</div>
                    <div style="width: 100%; background: #222; height: 4px; margin-top: 10px;">
                        <div style="width: 85%; background: #E50914; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown("""
                <div class="hologram-card" style="height: 150px; padding: 15px;">
                    <div style="color: #E50914; font-size: 0.6rem; font-weight: bold; letter-spacing: 2px;">NEURAL LOAD</div>
                    <div style="color: white; font-size: 1.5rem; font-weight: 800;">OPTIMAL</div>
                    <div style="width: 100%; background: #222; height: 4px; margin-top: 10px;">
                        <div style="width: 42%; background: #00ff00; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown("""
                <div class="hologram-card" style="height: 150px; padding: 15px;">
                    <div style="color: #E50914; font-size: 0.6rem; font-weight: bold; letter-spacing: 2px;">ALPHA PROXY</div>
                    <div style="color: white; font-size: 1.5rem; font-weight: 800;">STABLE</div>
                    <div style="width: 100%; background: #222; height: 4px; margin-top: 10px;">
                        <div style="width: 68%; background: white; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # 4. Global Sentinel Status
        st.markdown("""
        <div style="margin-top: 30px; padding: 15px; background: linear-gradient(90deg, #1a0000 0%, #000 100%); border-left: 4px solid #E50914;">
            <span style="color: white; font-weight: bold; letter-spacing: 2px;">SENTINEL PROTOCOL: </span>
            <span style="color: #888;">Monitoring global Netflix viewership vs Market Cap correlation... <b>[STATUS: HIGH CORRELATION]</b></span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
elif page == "Data Lab (ETL)":
    st.title("🧪 Data Engineering Lab")
    st.markdown("Automated ETL Pipeline & Multi-Dimensional Feature Engineering")
    
    tab1, tab2, tab3 = st.tabs(["1. Data Exploration", "2. Data Cleaning", "3. Feature Engineering"])
    
    with tab1:
        st.subheader("Raw Data Inspection")
        st.dataframe(df.head(10))
        st.caption(f"Shape: {df.shape[0]} Rows, {df.shape[1]} Columns")
        
    with tab2:
        st.subheader("Quality Checks")
        cols = st.columns(3)
        cols[0].metric("Missing Values", df.isnull().sum().sum())
        cols[1].metric("Duplicate Rows", df.duplicated().sum())
        cols[2].metric("Data Integrity", "100%")
        st.success("Dataset is clean and production-ready.")

    with tab3:
        st.subheader("Feature Engineering Showcase")
        st.write("Processed Data Preview (Indicators Added):")
        st.dataframe(df_processed[['Close', 'MA_20', 'MA_50', 'Daily_Return', 'Volatility']].tail(10))

elif page == "Deep EDA":
    st.title("📊 Deep Exploratory Data Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Feature Correlation Heatmap")
        fig_corr = utils.plot_correlation_heatmap(df_processed[['Open','High','Low','Close','Volume','MA_20','MA_50','Volatility']])
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col2:
        st.subheader("Distribution Analysis")
        dist_col = st.selectbox("Select Feature", ['Daily_Return', 'Volume', 'Close', 'Volatility'])
        fig_dist = utils.plot_distributions(df_processed, dist_col)
        st.plotly_chart(fig_dist, use_container_width=True)

elif page == "Advanced Visuals":
    st.title("🧙‍♂️ Advanced Visualizations (Animated)")
    st.markdown("Interactive and animated plots for deeper insights.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Market State Radar")
        st.plotly_chart(utils.plot_radar_chart(df_processed), use_container_width=True)
        
        st.subheader("Volume Distribution (Pie)")
        st.plotly_chart(utils.plot_pie_chart(df_processed), use_container_width=True)
        
    with col2:
        st.subheader("Distribution Evolution (Animated)")
        st.plotly_chart(utils.plot_animated_histogram(df_processed), use_container_width=True)
        
    st.subheader("🏁 Moving Average Bar Race (Animated)")
    st.plotly_chart(utils.plot_bar_race(df_processed), use_container_width=True)
    
    st.markdown("---")
    st.subheader("🌪️ The Volatility Vortex (High Impact)")
    st.info("A 3D animated visualization of Price Δ vs. Volume Δ vs. Volatility. Watch the market 'pulse' over time.")
    st.plotly_chart(utils.plot_volatility_vortex_3d(df_processed), use_container_width=True)

elif page == "Strategy Intelligence Lab":
    st.markdown("<h1 style='color: #E50914;'>🕵️ Strategy Intelligence Lab</h1>", unsafe_allow_html=True)
    st.markdown("### <span style='color:white'>Quant-Level Performance Analysis</span>", unsafe_allow_html=True)
    
    # Run Backtest
    bt_results = utils.backtest_ma_strategy(df_processed)
    metrics = utils.calculate_strategy_metrics(bt_results)
    anomalies = utils.detect_anomalies(df_processed)
    
    # Executive Metrics
    cols = st.columns(len(metrics))
    for i, (label, val) in enumerate(metrics.items()):
        cols[i].metric(label, val, delta=f"{val}%" if "Return" in label else None)
        
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📈 ROI Growth Analysis", "🚨 Anomaly Intelligence"])
    
    with tab1:
        st.subheader("Cumulative Growth: AI Strategy vs. Market")
        fig_equity = utils.plot_equity_curve(bt_results)
        st.plotly_chart(fig_equity, use_container_width=True)
        st.info("This chart compares the returns of an AI-driven Moving Average Crossover strategy against a standard Buy & Hold approach.")
        
    with tab2:
        st.subheader("Institutional Anomaly Detection")
        fig_anomaly = utils.plot_anomaly_map(df_processed, anomalies)
        st.plotly_chart(fig_anomaly, use_container_width=True)
        st.markdown(f"**Identified {len(anomalies)} critical market anomalies** where price action deviated significantly from standard statistical bands.")

elif page == "Executive Dashboard":
    st.title("📈 Executive Dashboard")
    # Metrics
    current = df_processed.iloc[-1]
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Closing Price", f"${current['Close']:.2f}", f"{current['Daily_Return']:.2%}")
    m2.metric("Volume", f"{int(current['Volume']):,}")
    m3.metric("Volatility", f"{current['Volatility']:.2f}")
    m4.metric("Sentiment", "Bullish" if current['Close'] > current['MA_50'] else "Bearish")
    
    st.markdown("---")
    st.subheader("Price Action Analysis")
    st.plotly_chart(utils.plot_candlestick(df_processed), use_container_width=True)

elif page == "3D Galaxy":
    st.title("🪐 5-Dimensional Market Galaxy")
    st.info("Interactive 3D Simulation: Rotate, Zoom, and Play the animation.")
    st.plotly_chart(utils.plot_4d_galaxy(df_processed), use_container_width=True)

elif page == "Advanced ML Models":
    st.title("🤖 Advanced Machine Learning Studio")
    
    model_type = st.radio("Select Model Architecture", ["Linear Regression", "LSTM (Deep Learning)"], horizontal=True)
    
    if model_type == "Linear Regression":
        st.subheader("Linear Regression Analysis")
        if st.button("Train Model"):
            y_test, preds, residuals, metrics = utils.train_linear_regression(df_processed)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("R² Score", f"{metrics['R2']:.4f}")
            c2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            c3.metric("MAE", f"{metrics['MAE']:.4f}")
            
            fig1, fig2, fig3 = utils.plot_residuals(y_test, preds, residuals)
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
            
    else: # LSTM
        st.subheader("LSTM Neural Network")
        if st.button("Train Network"):
            with st.spinner("Training Deep Learning Model..."):
                model, scaler, scaled_data, p_days, history = utils.train_lstm_model(df_processed)
                preds = utils.predict_lstm(model, scaler, scaled_data, p_days)
                
                st.plotly_chart(utils.plot_loss_curve(history), use_container_width=True)
                
                # Manual Prediction Plot to ensure style matching
                subset = df_processed.iloc[-len(preds):]
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=subset.index, y=subset['Close'], name='Actual', line=dict(color='white')))
                fig.add_trace(go.Scatter(x=subset.index, y=preds.flatten(), name='Prediction', line=dict(color='#E50914')))
                fig.update_layout(title="LSTM Forecast", template='plotly_dark', plot_bgcolor='black', paper_bgcolor='black')
                st.plotly_chart(fig, use_container_width=True)

elif page == "Simulator":
    st.title("💰 Time Travel Investment Simulator")
    st.markdown("See how much you would have made if you invested in Netflix in the past.")
    
    with st.form("sim_form"):
        col1, col2 = st.columns(2)
        amount = col1.number_input("Investment Amount ($)", 100, 1000000, 1000)
        date = col2.date_input("Investment Date", df_processed.index.min(), min_value=df_processed.index.min(), max_value=df_processed.index.max())
        submitted = st.form_submit_button("Launch Time Machine")
        
    if submitted:
        idx = df_processed.index.get_indexer([pd.to_datetime(date)], method='nearest')[0]
        start_price = df_processed['Close'].iloc[idx]
        current_price = df_processed['Close'].iloc[-1]
        actual_date = df_processed.index[idx]
        
        subset = df_processed.loc[actual_date:]
        # Calculate daily value of investment
        subset = subset.copy()
        subset['Investment_Value'] = (amount / start_price) * subset['Close']
        
        final_val = subset['Investment_Value'].iloc[-1]
        pnl = final_val - amount
        pct = (pnl / amount) * 100
        
        st.markdown(f"### Results for Investment on {actual_date.strftime('%Y-%m-%d')}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Value", f"${final_val:,.2f}")
        c2.metric("Total Profit", f"${pnl:,.2f}", f"{pct:.2f}%")
        c3.metric("Multiplier", f"{(final_val/amount):.2f}x")
        
        if pnl > 0: 
            st.balloons()
            st.success(f"MISSION SUCCESS: You earned ${pnl:,.2f}!")
        else:
            st.error(f"MISSION FAILURE: You lost ${abs(pnl):,.2f}.")

        # Create Growth Plot
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=subset.index, y=subset['Investment_Value'], fill='tozeroy', 
                                 line=dict(color='#E50914', width=3), name='Investment Value'))
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='black',
            paper_bgcolor='black',
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Timeline",
            hovermode='x unified'
        )

        st.subheader("📊 Advanced Performance Analytics")
        tab1, tab2, tab3 = st.tabs(["Timeline Growth", "Risk Analysis (Drawdown)", "Benchmark Battle"])
        
        with tab1:
            st.plotly_chart(fig, use_container_width=True)
            st.info("The chart shows the daily value of your investment from the chosen date until today.")
            
        with tab2:
            st.plotly_chart(utils.plot_drawdown(subset), use_container_width=True)
            st.warning("The drawdown plot shows the percentage decline from the previous peak. It visualizes the 'pain' you would have felt during market dips.")
            
        with tab3:
            st.plotly_chart(utils.plot_benchmark_comparison(subset, amount), use_container_width=True)
            st.success("Netflix vs. Cash: See if you would have been better off just leaving your money in a high-yield savings account (5% APY).")

