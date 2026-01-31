import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io
from scipy import optimize # Required for XIRR calculation

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SIP Backtest Pro", layout="wide")
st.title("💰 Mutual Fund SIP Backtester & Optimizer")

# --- HELPER FUNCTIONS ---

def get_csv_url(url):
    """Converts Google Sheet URL to CSV export URL"""
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

def calculate_xirr(dates, amounts):
    """
    Calculates XIRR (Extended Internal Rate of Return).
    dates: list of datetime objects
    amounts: list of cash flows (negative for investments, positive for current value)
    """
    if len(dates) != len(amounts):
        return 0.0
    
    # Calculate days from first transaction
    min_date = min(dates)
    days = [(d - min_date).days for d in dates]
    
    # Define XNPV function (Net Present Value at various rates)
    def xnpv(rate):
        # Prevent division by zero or negative rates issues
        if rate <= -1:
            return float('inf')
        return sum([a / ((1 + rate) ** (d / 365.0)) for a, d in zip(amounts, days)])

    try:
        # Newton-Raphson method to find the rate where NPV = 0
        return optimize.newton(xnpv, 0.1)
    except (RuntimeError, OverflowError):
        return 0.0

@st.cache_data
def load_data(sheet_url):
    """Robust data loader"""
    try:
        csv_url = get_csv_url(sheet_url)
        response = requests.get(csv_url)
        response.raise_for_status()
        
        if "<!DOCTYPE html>" in response.text[:100]:
            st.error("⚠️ Error: The Google Sheet is not public. Please change permissions to 'Anyone with the link'.")
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        date_col = next((col for col in df.columns if col.lower() == 'date'), None)
        if not date_col:
            st.error("⚠️ Error: Could not find a 'Date' column.")
            return pd.DataFrame()
            
        df.rename(columns={date_col: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        
        # Clean Numbers
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
        
    except Exception as e:
        st.error(f"⚠️ Error processing data: {str(e)}")
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
default_url = "https://docs.google.com/spreadsheets/d/18BIX9gGqIocnzACQk6pfAIb68-VhRpwOvX8Ni53el7I/edit?gid=0#gid=0"
sheet_url = st.sidebar.text_input("Paste Google Sheet URL", value=default_url)
sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", value=10000, step=1000)
years = st.sidebar.slider("Duration (Years)", 1, 10, 3)

# --- MAIN APP LOGIC ---
if sheet_url:
    df = load_data(sheet_url)
    
    if not df.empty:
        all_funds = df.columns.tolist()
        st.sidebar.markdown("---")
        selected_funds = st.sidebar.multiselect("Select Funds to Analyze", all_funds, default=all_funds[:2])
        
        if selected_funds:
            end_date = df.index.max()
            start_date = end_date - pd.DateOffset(years=years)
            df_filtered = df.loc[start_date:end_date, selected_funds]
            df_filtered = df_filtered.dropna(how='any')
            
            if df_filtered.empty:
                st.warning("⚠️ No data available for selected funds in this period.")
            else:
                st.success(f"✅ Analyzing {len(selected_funds)} funds | {len(df_filtered)} trading days")

                # --- 1. OPTIMIZATION ---
                st.subheader("1. AI Optimized Strategy")
                
                returns = df_filtered.pct_change()
                mean_returns = returns.mean() * 252
                cov_matrix = returns.cov() * 252
                
                num_portfolios = 2000
                results = np.zeros((3, num_portfolios))
                weights_record = []
                
                for i in range(num_portfolios):
                    weights = np.random.random(len(selected_funds))
                    weights /= np.sum(weights)
                    weights_record.append(weights)
                    p_ret = np.sum(mean_returns * weights)
                    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    results[0,i] = p_ret
                    results[1,i] = p_vol
                    results[2,i] = p_ret / p_vol

                best_idx = np.argmax(results[2])
                best_weights = weights_record[best_idx]
                best_allocation = dict(zip(selected_funds, best_weights))
                
                cols = st.columns(len(selected_funds))
                for i, (f, w) in enumerate(best_allocation.items()):
                    cols[i].metric(f, f"{w*100:.1f}%")

                # --- 2. BACKTEST WITH XIRR ---
                st.subheader("2. Backtest Results (XIRR)")
                monthly_data = df_filtered.resample('MS').first()
                total_invested = 0
                units = {f:0.0 for f in selected_funds}
                
                # We need to track cash flows for XIRR
                # Format: (Date, Amount). Amount is negative for SIP, positive for final value.
                cash_flows_date = []
                cash_flows_amount = []
                
                ledger = []
                
                for date, row in monthly_data.iterrows():
                    # 1. Record the SIP Investment (Outflow)
                    total_invested += sip_amount
                    cash_flows_date.append(date)
                    cash_flows_amount.append(-sip_amount)
                    
                    # 2. Buy Units
                    for f in selected_funds:
                        units[f] += (sip_amount * best_allocation[f]) / row[f]
                    
                    # 3. Track Portfolio Value
                    curr_val = sum([units[f] * row[f] for f in selected_funds])
                    ledger.append({'Date': date, 'Invested': total_invested, 'Value': curr_val})
                
                res_df = pd.DataFrame(ledger)
                
                if not res_df.empty:
                    final_val = res_df.iloc[-1]['Value']
                    invested = res_df.iloc[-1]['Invested']
                    profit = final_val - invested
                    
                    # Add final value as a positive cash flow on the last date
                    cash_flows_date.append(res_df.iloc[-1]['Date'])
                    cash_flows_amount.append(final_val)
                    
                    # Calculate XIRR
                    xirr_val = calculate_xirr(cash_flows_date, cash_flows_amount) * 100

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Invested", f"₹{invested:,.0f}")
                    c2.metric("Final Value", f"₹{final_val:,.0f}")
                    c3.metric("Net Profit", f"₹{profit:,.0f}")
                    c4.metric("XIRR", f"{xirr_val:.2f}%")
                    
                    fig = px.line(res_df, x='Date', y=['Invested', 'Value'], 
                                  color_discrete_map={'Invested':'#D3D3D3', 'Value':'#00CC96'})
                    fig.update_traces(fill='tozeroy')
                    st.plotly_chart(fig, use_container_width=True)
                    
        else:
            st.info("👈 Please select funds from the sidebar.")
