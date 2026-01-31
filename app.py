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
if sheet_url:
    with st.spinner("Processing all tabs in Google Sheet..."):
        df = load_data(sheet_url)
    
    if not df.empty:
        all_funds = df.columns.tolist()
        st.sidebar.markdown("---")
        selected_funds = st.sidebar.multiselect("Select Funds (Min 2)", all_funds, default=all_funds[:2])
        
        if len(selected_funds) > 1:
            # --- DATE ALIGNMENT LOGIC ---
            # 1. Select only the funds the user wants
            df_selected = df[selected_funds]
            
            # 2. Find the "Common Window"
            valid_data = df_selected.dropna()
            
            if valid_data.empty:
                st.error("❌ No overlapping dates found. These funds never traded on the same days.")
            else:
                common_start = valid_data.index.min()
                common_end = valid_data.index.max()
                
                # Calculate max years available in this common window
                max_available_years = (common_end - common_start).days / 365.25
                
                st.sidebar.markdown("---")
                st.sidebar.info(f"📅 **Common Data Found:**\n{common_start.date()} to {common_end.date()}")
                
                # Slider for User Duration
                if max_available_years < 0.1:
                     st.error("Less than 1 month of common data available.")
                else:
                    years = st.sidebar.slider(
                        "Analysis Duration", 
                        min_value=0.1, 
                        max_value=float(f"{max_available_years:.1f}"), 
                        value=float(f"{max_available_years:.1f}"),
                        step=0.1
                    )
                    
                    # --- DATE FIX: Use days instead of years to handle float values ---
                    days_offset = int(years * 365.25)
                    analysis_start = common_end - pd.DateOffset(days=days_offset)
                    
                    if analysis_start < common_start: analysis_start = common_start
                    
                    df_filtered = valid_data.loc[analysis_start:common_end]
                    
                    st.success(f"✅ Analyzing aligned data: **{analysis_start.date()}** to **{common_end.date()}** ({len(df_filtered)} days)")

                    # --- 2. MONTE CARLO ENGINE ---
                    st.subheader(f"1. AI Strategy: Maximizing {optimize_for}")
                    
                    daily_returns = df_filtered.pct_change().dropna()
                    mean_returns = daily_returns.mean() * 252 
                    cov_matrix = daily_returns.cov() * 252
                    
                    num_portfolios = 3000
                    results_list = []
                    all_weights = []
                    
                    progress_bar = st.progress(0)
                    for i in range(num_portfolios):
                        weights = np.random.random(len(selected_funds))
                        weights /= np.sum(weights)
                        all_weights.append(weights)
                        
                        # Return & Vol
                        port_return = np.sum(mean_returns * weights)
                        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        
                        # Sharpe
                        sharpe = port_return / port_volatility if port_volatility > 0 else 0
                        
                        # RoMaD
                        port_daily_ret = daily_returns.dot(weights)
                        cumulative_ret = (1 + port_daily_ret).cumprod()
                        peak = cumulative_ret.cummax()
                        drawdown = (cumulative_ret - peak) / peak
                        max_dd = abs(drawdown.min())
                        
                        romad = port_return / max_dd if max_dd > 0 else 0
                        
                        results_list.append({
                            "Return": port_return,
                            "Sharpe": sharpe,
                            "RoMaD": romad,
                            "MaxDD": max_dd
                        })
                        
                        if i % 300 == 0: progress_bar.progress(i/num_portfolios)
                    
                    progress_bar.empty()
                    results_df = pd.DataFrame(results_list)
                    
                    # Select Best
                    if "RoMaD" in optimize_for:
                        best_idx = results_df['RoMaD'].idxmax()
                    elif "Sharpe" in optimize_for:
                        best_idx = results_df['Sharpe'].idxmax()
                    else: 
                        best_idx = results_df['Return'].idxmax()

                    best_weights = all_weights[best_idx]
                    best_stats = results_df.iloc[best_idx]
                    
                    # Display Allocation
                    best_allocation = dict(zip(selected_funds, best_weights))
                    cols = st.columns(len(selected_funds))
                    for i, (f, w) in enumerate(best_allocation.items()):
                        cols[i].metric(f, f"{w*100:.1f}%")
                    
                    st.caption(f"Theoretical Stats: RoMaD: {best_stats['RoMaD']:.2f} | Sharpe: {best_stats['Sharpe']:.2f} | Est. Return: {best_stats['Return']*100:.1f}%")

                    # --- 3. SIP BACKTEST ---
                    st.subheader("2. Backtest Results (Realized)")
                    
                    monthly_data = df_filtered.resample('MS').first()
                    total_invested = 0
                    units = {f:0.0 for f in selected_funds}
                    cash_flows_date = []
                    cash_flows_amount = []
                    ledger = []
                    
                    for date, row in monthly_data.iterrows():
                        total_invested += sip_amount
                        cash_flows_date.append(date)
                        cash_flows_amount.append(-sip_amount)
                        
                        for f in selected_funds:
                            if row[f] > 0:
                                units[f] += (sip_amount * best_allocation[f]) / row[f]
                        
                        curr_val = sum([units[f] * row[f] for f in selected_funds])
                        ledger.append({'Date': date, 'Invested': total_invested, 'Value': curr_val})
                    
                    res_df = pd.DataFrame(ledger)
                    
                    if not res_df.empty:
                        final_val = res_df.iloc[-1]['Value']
                        invested = res_df.iloc[-1]['Invested']
                        profit = final_val - invested
                        
                        cash_flows_date.append(res_df.iloc[-1]['Date'])
                        cash_flows_amount.append(final_val)
                        
                        xirr_val = calculate_xirr(cash_flows_date, cash_flows_amount) * 100
                        
                        res_df['Peak'] = res_df['Value'].cummax()
                        res_df['Drawdown'] = (res_df['Value'] - res_df['Peak']) / res_df['Peak']
                        realized_max_dd = res_df['Drawdown'].min() * 100
                        
                        realized_romad = xirr_val / abs(realized_max_dd) if realized_max_dd != 0 else 0
                        
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Total Invested", f"₹{invested:,.0f}")
                        c2.metric("Current Value", f"₹{final_val:,.0f}")
                        c3.metric("XIRR", f"{xirr_val:.2f}%", delta="Net Profit")
                        c4.metric("Sharpe", f"{best_stats['Sharpe']:.2f}")
                        c5.metric("RoMaD", f"{realized_romad:.2f}", help=f"Max DD: {realized_max_dd:.1f}%")

                        fig = px.line(res_df, x='Date', y=['Invested', 'Value'], 
                                      title=f"Performance ({optimize_for} Strategy)",
                                      color_discrete_map={'Invested':'#D3D3D3', 'Value':'#00CC96'})
                        fig.update_traces(fill='tozeroy')
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("👈 Select at least 2 funds to start optimization.")
