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
            df_filtered = df.loc[start_date:end_date, selected_funds].dropna(how='any')
            
            if df_filtered.empty:
                st.warning("⚠️ No overlapping data found for these funds in this period.")
            else:
                st.success(f"✅ Analyzing {len(selected_funds)} funds | {len(df_filtered)} trading days")

                # --- 2. MONTE CARLO ENGINE ---
                st.subheader(f"1. AI Strategy: Maximizing {optimize_for}")
                
                daily_returns = df_filtered.pct_change().dropna()
                mean_returns = daily_returns.mean() * 252 
                cov_matrix = daily_returns.cov() * 252
                
                num_portfolios = 2000
                results_list = []
                all_weights = []
                
                # Simulation
                progress_bar = st.progress(0)
                for i in range(num_portfolios):
                    weights = np.random.random(len(selected_funds))
                    weights /= np.sum(weights)
                    all_weights.append(weights)
                    
                    # A. Return & Volatility
                    port_return = np.sum(mean_returns * weights)
                    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    
                    # B. Sharpe Ratio
                    sharpe = port_return / port_volatility if port_volatility > 0 else 0
                    
                    # C. Max Drawdown & RoMaD
                    # Calculate cumulative path for this specific weight mix
                    port_daily_ret = daily_returns.dot(weights)
                    cumulative_ret = (1 + port_daily_ret).cumprod()
                    peak = cumulative_ret.cummax()
                    drawdown = (cumulative_ret - peak) / peak
                    max_dd = abs(drawdown.min())
                    
                    romad = port_return / max_dd if max_dd > 0 else 0
                    
                    # Store everything
                    results_list.append({
                        "Return": port_return,
                        "Sharpe": sharpe,
                        "RoMaD": romad,
                        "MaxDD": max_dd
                    })
                    
                    if i % 200 == 0: progress_bar.progress(i/num_portfolios)
                
                progress_bar.empty()
                results_df = pd.DataFrame(results_list)
                
                # --- SELECT BEST PORTFOLIO BASED ON USER INPUT ---
                if "RoMaD" in optimize_for:
                    best_idx = results_df['RoMaD'].idxmax()
                elif "Sharpe" in optimize_for:
                    best_idx = results_df['Sharpe'].idxmax()
                else: # Returns
                    best_idx = results_df['Return'].idxmax()

                best_weights = all_weights[best_idx]
                best_stats = results_df.iloc[best_idx]
                
                # Display Allocation
                best_allocation = dict(zip(selected_funds, best_weights))
                cols = st.columns(len(selected_funds))
                for i, (f, w) in enumerate(best_allocation.items()):
                    cols[i].metric(f, f"{w*100:.1f}%")
                
                # Display Theoretical Stats of this Strategy
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
                    
                    # Final Inflow
                    cash_flows_date.append(res_df.iloc[-1]['Date'])
                    cash_flows_amount.append(final_val)
                    
                    # 1. XIRR
                    xirr_val = calculate_xirr(cash_flows_date, cash_flows_amount) * 100
                    
                    # 2. Realized Drawdown
                    res_df['Peak'] = res_df['Value'].cummax()
                    res_df['Drawdown'] = (res_df['Value'] - res_df['Peak']) / res_df['Peak']
                    realized_max_dd = res_df['Drawdown'].min() * 100
                    
                    # 3. Realized RoMaD & Sharpe (Approx)
                    realized_romad = xirr_val / abs(realized_max_dd) if realized_max_dd != 0 else 0
                    
                    # Display ALL Metrics
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Total Invested", f"₹{invested:,.0f}")
                    c2.metric("Current Value", f"₹{final_val:,.0f}")
                    c3.metric("XIRR", f"{xirr_val:.2f}%", delta="Net Profit")
                    
                    # Highlights
                    c4.metric("Sharpe Ratio", f"{best_stats['Sharpe']:.2f}", help="Return per unit of Volatility")
                    c5.metric("RoMaD", f"{realized_romad:.2f}", help=f"Return per unit of Crash. (Max DD: {realized_max_dd:.1f}%)")

                    # Chart
                    fig = px.line(res_df, x='Date', y=['Invested', 'Value'], 
                                  title=f"Performance ({optimize_for} Strategy)",
                                  color_discrete_map={'Invested':'#D3D3D3', 'Value':'#00CC96'})
                    fig.update_traces(fill='tozeroy')
                    st.plotly_chart(fig, use_container_width=True)
                    
        else:
            st.warning("👈 Select at least 2 funds to start optimization.")
