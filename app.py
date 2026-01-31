import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
import re
from scipy import optimize

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro Strategy Lab", layout="wide")
st.title("🧪 Multi-Sheet Strategy Lab")
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

def get_multisheet_url(url):
    """
    Extracts the ID from a Google Sheet URL and creates an export link 
    that downloads the entire workbook (all tabs) at once.
    """
    if "docs.google.com/spreadsheets" not in url:
        return url
    
    # Robust Regex to find the Sheet ID regardless of URL format
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    if match:
        sheet_id = match.group(1)
        # We use 'format=xlsx' because it is the ONLY way to fetch ALL tabs in one request.
        return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    
    return url

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
        # Convert Google Sheet URL to a Workbook Export URL
        export_url = get_multisheet_url(sheet_url)
        
        # Fetch the data
        response = requests.get(export_url, timeout=30)
        response.raise_for_status()
        
        # Check if permissions are wrong
        if "<!DOCTYPE html>" in response.text[:100]:
            st.error("⚠️ Error: Sheet is private. Please click 'Share' > 'Anyone with the link'.")
            return pd.DataFrame()

        # Load the entire workbook (sheet_name=None loads ALL tabs)
        all_sheets = pd.read_excel(io.BytesIO(response.content), sheet_name=None)
        
        processed_dfs = []
        
        # Iterate through every tab in the Google Sheet
        for sheet_name, df in all_sheets.items():
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            # Look for a Date column
            date_col = next((col for col in df.columns if col.lower() == 'date'), None)
            
            if date_col:
                # Standardize Date
                df.rename(columns={date_col: 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                
                # --- FIX FOR "Reindexing only valid with uniquely valued Index objects" ---
                # 1. Drop rows where Date is missing
                df = df.dropna(subset=['Date'])
                # 2. Drop DUPLICATE dates (Keep the first occurrence)
                df = df.drop_duplicates(subset=['Date'], keep='first')
                
                df.set_index('Date', inplace=True)
                
                # Clean Numbers (Remove commas, quotes)
                for col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove empty rows/cols
                df = df.dropna(how='all')
                
                # Add to our list of dataframes
                processed_dfs.append(df)
        
        if not processed_dfs:
            st.error("⚠️ Could not find a 'Date' column in any tab of your Google Sheet.")
            return pd.DataFrame()

        # Merge all tabs into one massive table based on Date
        full_df = pd.concat(processed_dfs, axis=1)
        
        # Remove duplicate columns (if the same fund appears in multiple tabs)
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
        
        return full_df

    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
default_url = "https://docs.google.com/spreadsheets/d/18BIX9gGqIocnzACQk6pfAIb68-VhRpwOvX8Ni53el7I/edit?gid=0#gid=0"
sheet_url = st.sidebar.text_input("Paste Google Sheet URL", value=default_url)
st.sidebar.caption("✅ Reads ALL tabs from your sheet automatically")

sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", value=10000, step=1000)

st.sidebar.subheader("🎯 Optimization Goal")
optimize_for = st.sidebar.selectbox(
    "Maximize Strategy For:",
    ("RoMaD (Safety)", "Sharpe Ratio (Efficiency)", "Returns (High Growth)")
)

# --- MAIN APP LOGIC ---
if sheet
