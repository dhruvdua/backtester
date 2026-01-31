import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
from scipy import optimize

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Strategy Lab", layout="wide")
st.title("🧪 Investment Strategy Lab")
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_csv_url(url):
    if "docs.google.com/spreadsheets" not in url: return url
    base_url = url.split('/edit')[0]
    gid = "0"
    if "gid=" in url:
        import re
        gid_match = re.search(r'gid=(\d+)', url)
        if gid_match: gid = gid_match.group(1)
    return f"{base_url}/export?format=csv&gid={gid}"

def calculate_xirr(dates, amounts):
    if len(dates) != len(amounts): return 0.0
    min_date = min(dates)
    days = [(d - min_date).days for d in dates]
    def xnpv(rate):
        if rate <= -1: return float('inf')
        return sum([a / ((1 + rate) ** (d / 365.0)) for a, d in zip(amounts, days)])
    try:
        return optimize.newton(xnpv, 0.1)
    except (RuntimeError, OverflowError):
        return 0.0

@st.cache_data
def load_data(sheet_url):
    try:
        csv_url = get_csv_url(sheet_url)
        response = requests.get(csv_url, timeout=10)
        response.raise_for_status()
        
        if "<!DOCTYPE html>" in response.text[:100]:
            st.error("⚠️ Error: Sheet is private. Set access to 'Anyone with the link'.")
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        date_col = next((col for col in df.columns if col.lower() == 'date'), None)
        if not date_col:
            st.error("⚠️ Error: No 'Date' column found.")
            return pd.DataFrame()
            
        df.rename(columns={date_col: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
default_url = "https://docs.google.com/spreadsheets/d/18BIX9gGqIocnzACQk6pfAIb68-VhRpwOvX8Ni53el7I/edit?gid=0#gid=0"
sheet_url = st.sidebar.text_input("Paste Google Sheet URL", value=default_url)
sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", value=10000, step=1000)
years = st.sidebar.slider("Duration (Years)", 1, 10, 3)

st.sidebar.subheader("🎯 Optimization Goal")
# User chooses what to maximize
optimize_for = st.sidebar.selectbox(
    "Maximize Strategy For:",
    ("RoMaD (Safety)", "Sharpe Ratio (Efficiency)", "Returns (High Growth)")
)

# --- MAIN APP LOGIC ---
if sheet_url:
    df = load_data(sheet_url)
    
    if not df.empty:
        all_funds = df.columns.tolist()
        st.sidebar.markdown("---")
        selected_funds = st.sidebar.multiselect("Select Funds (Min 2)", all_funds, default=all_funds[:2])
        
        if len(selected_funds) > 1:
            # 1. Slice Data
            end_date = df.index.max()
            start_date = end_date - pd.DateOffset(years=years)
            df_filtered = df.loc[start_
