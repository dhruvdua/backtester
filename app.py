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
            df.columns = df.columns.astype(str).str.strip()
            date_col = next((col for col in df.columns if col.lower() == 'date'), None)
            if date_col:
                df.rename(columns={date_col: 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['Date'])
                df = df.drop_duplicates(subset=['Date'], keep='first')
                df.set_index('Date', inplace=True)
                for col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(how='all')
                processed_dfs.append(df)
        
        if not processed_dfs: return pd.DataFrame()
        full_df = pd.concat(processed_dfs, axis=1)
        full_df = full_df.loc[:, ~full_df.columns.duplicated()]
        return full_df
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
default_url = "https://docs.google.com/spreadsheets/d/18BIX9gGqIocnzACQk6pfAIb68-VhRpwOvX8Ni53el7I/edit?gid=0#gid=0"
sheet_url = st.sidebar.text_input("Paste Google Sheet URL", value=default_url)
sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", value=10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Simulation Settings")
num_simulations = st.sidebar.number_input("Simulations", 500, 50000, 3000, step=500)

with st.sidebar.expander("🔎 Advanced Filters", expanded=False):
    st.caption("Restrict AI to portfolios that fit these criteria:")
    filter_romad = st.slider("RoMaD Range", 0.0, 5.0, (0.0, 5.0), step=0.1)
    filter_return = st.slider("Annual Return % (XIRR Proxy)", 0.0, 100.0, (0.0, 100.0), step=1.0)
    filter_rolling = st.slider("Avg 3Y Rolling %", 0.0, 100.0, (0.0, 100.0), step=1.0)
    filter_sharpe = st.slider("Sharpe Ratio", 0.0, 5.0, (0.0, 5.0), step=0.1)

# --- MAIN APP ---
if sheet_url:
    with st.spinner("Fetching data..."):
        df = load_data(sheet_url)
    
    if not df.empty:
        all_funds = df.columns.tolist()
        st.subheader("1. Select Funds & Time Period")
        c1, c2 = st.columns([1, 1])
        with c1:
            selected_funds = st.multiselect("Select Funds (Min 2)", all_funds, default=all_funds[:2])
        
        if len(selected_funds) > 1:
            df_selected = df[selected_funds]
            valid_data = df_selected.dropna()
            
            if valid_data.empty:
                st.error("❌ No overlapping dates found.")
            else:
                common_start = valid_data.index.min()
                common_end = valid_data.index.max()
                max_yrs = (common_end - common_start).days / 365.25
                
                with c2:
                    if max_yrs < 0.1:
                         st.error("Data < 1 month.")
                    else:
                        years = st.slider("Analysis Duration (Years)", 0.1, float(f"{max_yrs:.1f}"), float(f"{max_yrs:.1f}"), step=0.1)
                        days_offset = int(years * 365.25)
                        analysis_start = common_end - pd.DateOffset(days=days_offset)
                        if analysis_start < common_start: analysis_start = common_start
                        df_filtered = valid_data.loc[analysis_start:common_end]

                        # --- MANUAL ALLOCATION ---
                        st.markdown("---")
                        st.subheader("2. Define Your Strategy")
                        col_manual, col_chart = st.columns([1, 2])
                        manual_weights = {}
                        with col_manual:
                            st.write("#### 🎛️ Manual Allocation")
                            total_w = 0
                            for fund in selected_funds:
                                default_w = int(100/len(selected_funds))
                                w = st.number_input(f"{fund} (%)", min_value=0, max_value=100, value=default_w, step=5)
                                manual_weights[fund] = w
                                total_w += w
                            if total_w != 100: st.warning(f"Total: {total_w}%")

                        m_metrics, m_df = run_backtest(df_filtered, manual_weights, sip_amount)

                        with col_chart:
                            st.write("#### 📊 Your Performance")
                            if m_metrics:
                                mc1, mc2, mc3, mc4 = st.columns(4)
                                mc1.metric("XIRR", f"{m_metrics['XIRR']:.2f}%")
                                mc2.metric("3Y Rolling", f"{m_metrics['Rolling3Y']:.2f}%")
                                mc3.metric("RoMaD", f"{m_metrics['RoMaD']:.2f}")
                                mc4.metric("Sharpe", f"{m_metrics['Sharpe']:.2f}")
                                fig = px.line(m_df, x='Date', y=['Invested', 'Value'], 
                                              color_discrete_map={'Invested':'#D3D3D3', 'Value':'#3b82f6'})
                                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)

                        # --- AI OPTIMIZATION ---
                        st.markdown("---")
                        st.subheader("3. AI Optimization Engine")
                        
                        opt_col1, opt_col2 = st.columns([1, 3])
                        with opt_col1:
                            optimize_for = st.selectbox(
                                "Maximize For:",
                                ("Rolling Returns (Consistency)", "RoMaD (Max Safety)", "Sharpe (Max Efficiency)", "Returns (Max Profit)")
                            )
                            # --- NEW FEATURE: BEST X FUNDS ---
                            st.write("##### 🛠️ Best X Funds Strategy")
                            use_best_x = st.checkbox("Pick Subset of Funds?", value=False)
                            max_funds_pick = len(selected_funds)
                            if use_best_x:
                                x_funds = st.slider("Number of funds to pick (X)", 2, len(selected_funds), 2)
                            else:
                                x_funds = len(selected_funds)

                            run_opt = st.button("🚀 Run AI Optimizer")

                        if run_opt:
                            with st.spinner(f"Simulating {num_simulations} portfolios (Picking best {x_funds} funds)..."):
                                daily_returns = df_filtered.pct_change().dropna()
                                mean_returns = daily_returns.mean() * 252 
                                cov_matrix = daily_returns.cov() * 252
                                sim_data = []
                                roll_window = 756
                                
                                for i in range(num_simulations):
                                    w = np.random.random(len(selected_funds))
                                    
                                    # --- BEST X LOGIC ---
                                    if use_best_x:
                                        # Find indices of the 'x' largest random weights
                                        # partition moves the x-th smallest elements to front, so we want the end
                                        top_indices = np.argpartition(w, -x_funds)[-x_funds:]
                                        # Create a mask of zeros
                                        mask = np.zeros_like(w)
                                        # Set only top indices to match original random weights
                                        mask[top_indices] = w[top_indices]
                                        w = mask # Apply mask
                                    
                                    # Normalize so sum is 1.0 (100%)
                                    w /= np.sum(w)
                                    
                                    # Metrics
                                    ret = np.sum(mean_returns * w)
                                    vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                                    sharpe = ret / vol if vol > 0 else 0
                                    path_daily = daily_returns.dot(w)
                                    cum_path = (1 + path_daily).cumprod()
                                    dd = (cum_path - cum_path.cummax()) / cum_path.cummax()
                                    max_dd = abs(dd.min())
                                    romad = ret / max_dd if max_dd > 0 else 0
                                    
                                    avg_roll = 0
                                    if len(cum_path) > roll_window:
                                        roll_ret_series = (cum_path.values[roll_window:] / cum_path.values[:-roll_window]) ** (1/3) - 1
                                        avg_roll = np.mean(roll_ret_series)
                                    
                                    sim_data.append({
                                        "weights": w,
                                        "Returns": ret * 100,
                                        "RoMaD": romad,
                                        "Rolling": avg_roll * 100,
                                        "Sharpe": sharpe
                                    })
                                
                                sim_df = pd.DataFrame(sim_data)
                                mask = (
                                    (sim_df["RoMaD"].between(filter_romad[0], filter_romad[1])) &
                                    (sim_df["Returns"].between(filter_return[0], filter_return[1])) &
                                    (sim_df["Rolling"].between(filter_rolling[0], filter_rolling[1])) &
                                    (sim_df["Sharpe"].between(filter_sharpe[0], filter_sharpe[1]))
                                )
                                filtered_df = sim_df[mask]
                                
                                if filtered_df.empty:
                                    st.warning("⚠️ No portfolios matched filters. Showing best Unfiltered result.")
                                    final_df = sim_df
                                else:
                                    st.success(f"✅ Found {len(filtered_df)} portfolios.")
                                    final_df = filtered_df
                                
                                if "RoMaD" in optimize_for: best_row = final_df.loc[final_df['RoMaD'].idxmax()]
                                elif "Sharpe" in optimize_for: best_row = final_df.loc[final_df['Sharpe'].idxmax()]
                                elif "Rolling" in optimize_for: best_row = final_df.loc[final_df['Rolling'].idxmax()]
                                else: best_row = final_df.loc[final_df['Returns'].idxmax()]
                                
                                best_weights = {f: w*100 for f, w in zip(selected_funds, best_row['weights'])}
                                ai_metrics, ai_df = run_backtest(df_filtered, best_weights, sip_amount)
                                
                                comp_col1, comp_col2 = st.columns(2)
                                with comp_col1:
                                    st.write("**Manual Allocation**")
                                    man_df = pd.DataFrame.from_dict(manual_weights, orient='index', columns=['Weight'])
                                    st.dataframe(man_df.style.format("{:.1f}%"))
                                with comp_col2:
                                    st.write(f"**AI 'Best {x_funds}' Allocation**")
                                    # Highlight non-zero funds
                                    ai_alloc_df = pd.DataFrame.from_dict(best_weights, orient='index', columns=['Weight'])
                                    st.dataframe(ai_alloc_df.style.format("{:.1f}%"))
                                
                                comp_metrics = pd.DataFrame({
                                    "Metric": ["XIRR", "Avg 3Y Rolling", "RoMaD", "Max Drawdown", "Sharpe", "Final Value"],
                                    "Your Strategy": [
                                        f"{m_metrics['XIRR']:.2f}%", f"{m_metrics['Rolling3Y']:.2f}%", f"{m_metrics['RoMaD']:.2f}", 
                                        f"{m_metrics['MaxDD']:.2f}%", f"{m_metrics['Sharpe']:.2f}", f"₹{m_metrics['Current Value']:,.0f}"
                                    ],
                                    "AI Strategy": [
                                        f"{ai_metrics['XIRR']:.2f}%", f"{ai_metrics['Rolling3Y']:.2f}%", f"{ai_metrics['RoMaD']:.2f}", 
                                        f"{ai_metrics['MaxDD']:.2f}%", f"{ai_metrics['Sharpe']:.2f}", f"₹{ai_metrics['Current Value']:,.0f}"
                                    ]
                                })
                                st.table(comp_metrics)
                                merged_chart = m_df[['Date', 'Value']].rename(columns={'Value': 'Your Strategy'})
                                merged_chart['AI Strategy'] = ai_df['Value']
                                merged_chart['Invested'] = m_df['Invested']
                                fig_comp = px.line(merged_chart, x='Date', y=['Invested', 'Your Strategy', 'AI Strategy'],
                                                   color_discrete_map={'Invested':'#D3D3D3', 'Your Strategy':'#3b82f6', 'AI Strategy':'#22c55e'})
                                st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("👈 Select at least 2 funds from the sidebar to begin.")
