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
st.title("🧪 Investment Strategy Lab")
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_multisheet_url(url):
    if "docs.google.com/spreadsheets" not in url: return url
    match = re.search(r'/d/([a-zA-Z0-9-_]+)', url)
    if match:
        return f"https://docs.google.com/spreadsheets/d/{match.group(1)}/export?format=xlsx"
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

def calculate_rolling_returns(series, years=3):
    trading_days = int(years * 252)
    if len(series) < trading_days: return 0.0
    rolling_ret = (series / series.shift(trading_days)) ** (1/years) - 1
    return rolling_ret.mean()

def run_backtest(df, weights, sip_amount):
    monthly_data = df.resample('MS').first()
    total_invested = 0
    units = {f:0.0 for f in weights.keys()}
    cash_flows_date = []
    cash_flows_amount = []
    ledger = []
    
    for date, row in monthly_data.iterrows():
        total_invested += sip_amount
        cash_flows_date.append(date)
        cash_flows_amount.append(-sip_amount)
        for f in weights.keys():
            if row[f] > 0:
                allocation_amt = sip_amount * (weights[f] / 100.0)
                units[f] += allocation_amt / row[f]
        curr_val = sum([units[f] * row[f] for f in weights.keys()])
        ledger.append({'Date': date, 'Invested': total_invested, 'Value': curr_val})
    
    res_df = pd.DataFrame(ledger)
    if res_df.empty: return None, None
    
    daily_ret = df.pct_change().dropna()
    w_array = np.array([weights.get(col, 0) for col in df.columns]) / 100.0
    daily_port_ret = daily_ret.dot(w_array)
    daily_cum_path = (1 + daily_port_ret).cumprod()
    
    final_val = res_df.iloc[-1]['Value']
    invested = res_df.iloc[-1]['Invested']
    
    cash_flows_date.append(res_df.iloc[-1]['Date'])
    cash_flows_amount.append(final_val)
    xirr_val = calculate_xirr(cash_flows_date, cash_flows_amount) * 100
    
    res_df['Peak'] = res_df['Value'].cummax()
    res_df['Drawdown'] = (res_df['Value'] - res_df['Peak']) / res_df['Peak']
    max_dd = abs(res_df['Drawdown'].min()) * 100
    romad = xirr_val / max_dd if max_dd > 0 else 0
    
    sharpe = (daily_port_ret.mean() * 252) / (daily_port_ret.std() * np.sqrt(252))
    avg_rolling_3y = calculate_rolling_returns(daily_cum_path, years=3) * 100

    metrics = {
        "Invested": invested,
        "Current Value": final_val,
        "XIRR": xirr_val,
        "RoMaD": romad,
        "MaxDD": max_dd,
        "Sharpe": sharpe,
        "Rolling3Y": avg_rolling_3y
    }
    return metrics, res_df

@st.cache_data
def load_data(sheet_url):
    try:
        export_url = get_multisheet_url(sheet_url)
        response = requests.get(export_url, timeout=30)
        response.raise_for_status()
        if "<!DOCTYPE html>" in response.text[:100]:
            st.error("⚠️ Error: Sheet is private.")
            return pd.DataFrame()

        all_sheets = pd.read_excel(io.BytesIO(response.content), sheet_name=None)
        processed_dfs = []
        for sheet_name, df in all_sheets.items():
            df.
