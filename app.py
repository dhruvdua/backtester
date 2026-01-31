import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import traceback

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Deep Debugger", layout="wide")
st.title("🛠️ Deep Debug Mode")

st.write("### 1. System Check")
st.write("✅ Imports successful. Streamlit is running.")

# --- HELPER FUNCTIONS ---
def get_csv_url(url):
    if "docs.google.com/spreadsheets" not in url:
        return url
    base_url = url.split('/edit')[0]
    gid = "0"
    if "gid=" in url:
        import re
        gid_match = re.search(r'gid=(\d+)', url)
        if gid_match:
            gid = gid_match.group(1)
    return f"{base_url}/export?format=csv&gid={gid}"

def load_data_debug(sheet_url):
    st.write("### 2. Data Loading Process")
    
    # A. URL Conversion
    csv_url = get_csv_url(sheet_url)
    st.write(f"**Step A:** Converting URL...")
    st.code(f"Original: {sheet_url}\nConverted: {csv_url}")
    
    try:
        # B. HTTP Request
        st.write("**Step B:** Sending HTTP Request to Google...")
        response = requests.get(csv_url, timeout=10)
        st.write(f"-> Status Code: `{response.status_code}`")
        
        if response.status_code != 200:
            st.error(f"❌ Failed to fetch data. Status code: {response.status_code}")
            return pd.DataFrame()
        
        # C. Inspect Raw Content
        raw_text = response.text
        st.write("**Step C:** Inspecting first 200 characters of response:")
        st.code(raw_text[:200])
        
        if "<!DOCTYPE html>" in raw_text:
            st.error("❌ ERROR: Received HTML (Login Page) instead of CSV. Your Google Sheet is not public.")
            st.info("Fix: Go to Sheet > Share > General Access > Change 'Restricted' to 'Anyone with the link'.")
            return pd.DataFrame()
            
        # D. Parse CSV
        st.write("**Step D:** Parsing CSV data...")
        df = pd.read_csv(io.StringIO(raw_text), on_bad_lines='skip')
        st.write(f"-> Raw DataFrame shape: {df.shape}")
        st.dataframe(df.head(3))
        
        if df.empty:
            st.error("❌ ERROR: Pandas read the CSV but found no data (Empty DataFrame).")
            return pd.DataFrame()

        # E. Clean Columns
        st.write("**Step E:** Cleaning Columns...")
        df.columns = df.columns.str.strip()
        st.write(f"-> Columns found: `{df.columns.tolist()}`")
        
        # F. Find Date Column
        date_col = next((col for col in df.columns if col.lower() == 'date'), None)
        if not date_col:
            st.error(f"❌ ERROR: Could not find 'Date' column. Is it named differently?")
            return pd.DataFrame()
        
        st.write(f"-> Found Date column: `{date_col}`")
        df.rename(columns={date_col: 'Date'}, inplace=True)
        
        # G. Date Conversion
        st.write("**Step G:** Converting Dates (dayfirst=True)...")
        # Show sample before conversion
        st.write("-> Sample dates before conversion:", df['Date'].head(3).tolist())
        
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # Check for failures
        na_count = df['Date'].isna().sum()
        st.write(f"-> Rows with invalid dates (NaT): {na_count}")
        
        if na_count == len(df):
            st.error("❌ ERROR: All dates failed to convert. Check date format in Sheet.")
            return pd.DataFrame()
            
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        
        # H. Numeric Conversion
        st.write("**Step H:** Converting Numbers...")
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna()
        st.success(f"✅ Data Loaded Successfully! Final Shape: {df.shape}")
        return df

    except Exception as e:
        st.error("❌ CRITICAL EXCEPTION OCCURRED")
        st.code(traceback.format_exc())
        return pd.DataFrame()

# --- INPUTS ---
default_url = "https://docs.google.com/spreadsheets/d/1lMNqoG0Z_WehJVUlP4-3Jf6cDofjUOu5FMK-xJ1EnOU/edit?gid=0#gid=0"
sheet_url = st.text_input("Paste Google Sheet URL", value=default_url)

if st.button("🔴 RUN DEBUG"):
    df = load_data_debug(sheet_url)
    
    if not df.empty:
        st.write("### 3. App Logic Starting...")
        all_funds = df.columns.tolist()
        st.write(f"-> Funds Found: {all_funds}")
        
        selected_funds = st.multiselect("Select Funds", all_funds, default=all_funds[:2])
        st.write(f"-> User Selected: {selected_funds}")
