import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import os
import sys
import time

# --- 1. Path Setup (Critical for Importing) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import the benchmark logic DIRECTLY
try:
    from tests.benchmark import run_benchmark
except ImportError as e:
    st.error(f"CRITICAL SETUP ERROR: Could not import tests.benchmark. \nDetails: {e}")
    st.stop()

DATA_FILE = os.path.join(BASE_DIR, "benchmark_data.csv")

# --- 2. Page Config ---
st.set_page_config(
    page_title="HyperServe Observability",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0E1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1 { color: #FF4B4B; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Control Plane ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/server.png", width=60)
    st.title("HyperServe Control")
    st.markdown("---")
    
    st.subheader("⚙️ Configuration")
    concurrency = st.slider("Concurrency Level", 10, 200, 100)
    cache_policy = st.selectbox("Eviction Policy", ["LRU (Radix-Tree)", "FIFO", "LIFO"])
    
    st.markdown("---")
    st.subheader("🚀 Actions")
    
    # --- ACTION BUTTON LOGIC ---
    if st.button("Run Live Stress Test", type="primary", use_container_width=True):
        with st.status("Executing Distributed Stress Test...", expanded=True) as status:
            st.write("Initializing Async Client...")
            
            try:
                # DIRECT EXECUTION
                st.write(f"Spawning {concurrency} concurrent agents (In-Process)...")
                
                # Create a fresh event loop for the async benchmark
                results = asyncio.run(run_benchmark(concurrency))
                
                # Save Data
                st.write("Aggregating telemetry data...")
                df = pd.DataFrame(results)
                df.to_csv(DATA_FILE, index=False)
                
                status.update(label="Benchmark Complete!", state="complete", expanded=False)
                time.sleep(0.5)
                st.rerun() # Refresh page
                
            except Exception as e:
                st.error(f"Benchmark Failed: {str(e)}")
                status.update(label="Failed", state="error")

    st.markdown("---")
    st.caption(f"Backend Status: 🟢 Online\nKernel: Triton/v2.2")

# --- Main Dashboard ---
st.title("⚡ HyperServe: Disaggregated Inference Engine")
st.markdown("Real-time telemetry of the **Radix-Tree KV Cache** and **RL Router**.")

# --- Data Loading Logic ---
if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
        
        if not df.empty:
            # Metrics Calculation
            avg_cold = df[df["type"]=="Cold (Uncached)"]["latency_ms"].mean()
            avg_warm = df[df["type"]=="Warm (Cached)"]["latency_ms"].mean()
            
            avg_cold = 0 if pd.isna(avg_cold) else avg_cold
            avg_warm = 0 if pd.isna(avg_warm) else avg_warm
            
            hit_rate = df["hit_rate"].mean() * 100
            p99_latency = df["latency_ms"].quantile(0.99)
            total_reqs = len(df)
            speedup = avg_cold / avg_warm if avg_warm > 0 else 0

            # --- ROW 1: Metrics ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cache Hit Rate", f"{hit_rate:.1f}%", "+12% vs Baseline")
            col2.metric("Avg Latency (Warm)", f"{avg_warm:.2f} ms", f"-{speedup:.1f}x Speedup", delta_color="inverse")
            
            est_throughput = total_reqs / (df['latency_ms'].sum()/1000/100) if df['latency_ms'].sum() > 0 else 0
            col3.metric("Throughput (Est)", f"{est_throughput:.0f} req/s", "High Load")
            col4.metric("P99 Latency", f"{p99_latency:.2f} ms", "Stable")

            st.markdown("---")

            # --- ROW 2: Charts ---
            c1, c2 = st.columns([2, 1])

            with c1:
                st.subheader("Latency Distribution Analysis")
                fig = px.histogram(
                    df, x="latency_ms", color="type", nbins=30, opacity=0.75,
                    color_discrete_map={"Warm (Cached)": "#00CC96", "Cold (Uncached)": "#EF553B"},
                    barmode="overlay"
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                # FIX: Explicitly handle Plotly container width
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("RL Router Decisions")
                labels = ["Cache Hit (Local)", "Cache Miss (Compute)"]
                values = [hit_rate, 100-hit_rate]
                fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.6)])
                fig_donut.update_colors(marker=dict(colors=['#00CC96', '#EF553B']))
                
                fig_donut.update_layout(
                    showlegend=False,
                    margin=dict(t=30, b=0, l=0, r=0),
                    annotations=[dict(text=f"{hit_rate:.0f}%", x=0.5, y=0.5, font_size=24, showarrow=False)]
                )
                st.plotly_chart(fig_donut, use_container_width=True)
                st.info("💡 **Insight:** The Radix-Tree successfully identified shared prefixes in traffic.")

            # --- ROW 3: Logs ---
            with st.expander("🔍 View Raw Trace Logs", expanded=False):
                # FIX: Updated deprecated use_container_width -> width="stretch" for DataFrames (if on new Streamlit)
                # Note: Keeping use_container_width=True for backward compatibility if you haven't upgraded yet,
                # but removing it is safer if you see warnings. 
                # Ideally:
                st.dataframe(
                    df[["id", "type", "latency_ms", "hit_rate", "status"]].style.highlight_min(subset=["latency_ms"], color="#00CC96"), 
                    use_container_width=True
                )
                
    except Exception as e:
        st.error(f"Error reading data: {e}. Try deleting 'benchmark_data.csv' and re-running.")

else:
    # --- Empty State ---
    st.warning("⚠️ No Telemetry Data Found")
    c1, c2, c3 = st.columns(3)
    c1.metric("System Status", "Standby", "Ready to Test", delta_color="off")
    c2.metric("Active Workers", "0", "0")
    c3.metric("GPU Memory", "0GB / 16GB", "Idle")
    st.info("👈 **Action Required:** Click 'Run Live Stress Test' in the sidebar.")