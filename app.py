import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import io
import re
from scipy import optimize
from datetime import date

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
    
    # Portfolio Volatility
    port_vol = daily_port_ret.std() * np.sqrt(252)
    sharpe = (daily_port_ret.mean() * 252) / port_vol if port_vol > 0 else 0
    
    # Diversification Ratio
    individual_vols = daily_ret.std() * np.sqrt(252)
    weighted_vol = np.sum(w_array * individual_vols)
    div_ratio = weighted_vol / port_vol if port_vol > 0 else 0
    
    avg_rolling_3y = calculate_rolling_returns(daily_cum_path, years=3) * 100

    metrics = {
        "Invested": invested,
        "Current Value": final_val,
        "XIRR": xirr_val,
        "RoMaD": romad,
        "MaxDD": max_dd,
        "Sharpe": sharpe,
        "Rolling3Y": avg_rolling_3y,
        "DivRatio": div_ratio
    }
    return metrics, res_df

@st.cache_data
def load_data_google(sheet_url):
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

# --- AMFI API HELPERS ---
@st.cache_data
def get_all_amfi_schemes():
    url = "https://api.mfapi.in/mf"
    try:
        response = requests.get(url)
        return pd.DataFrame(response.json())
    except:
        return pd.DataFrame()

@st.cache_data
def fetch_amfi_nav(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        response = requests.get(url)
        data = response.json()
        if data['status'] == 'SUCCESS':
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'])
            df.set_index('date', inplace=True)
            df.sort_index(ascending=True, inplace=True)
            df.rename(columns={'nav': data['meta']['scheme_name']}, inplace=True)
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Configuration")
data_source = st.sidebar.radio("Data Source:", ("Google Sheet", "AMFI API (Live)"))

if 'amfi_data' not in st.session_state:
    st.session_state['amfi_data'] = pd.DataFrame()

df_global = pd.DataFrame()

if data_source == "Google Sheet":
    default_url = "https://docs.google.com/spreadsheets/d/18BIX9gGqIocnzACQk6pfAIb68-VhRpwOvX8Ni53el7I/edit?gid=0#gid=0"
    sheet_url = st.sidebar.text_input("Paste Google Sheet URL", value=default_url)
    if sheet_url:
        with st.spinner("Fetching Google Sheet..."):
            df_global = load_data_google(sheet_url)

elif data_source == "AMFI API (Live)":
    st.sidebar.info("Search & Add funds directly from AMFI.")
    schemes_df = get_all_amfi_schemes()
    if not schemes_df.empty:
        scheme_options = schemes_df['schemeCode'].astype(str) + " - " + schemes_df['schemeName']
        selected_schemes = st.sidebar.multiselect("Search Mutual Funds:", scheme_options)
        
        if st.sidebar.button("Fetch Data"):
            if selected_schemes:
                dfs = []
                progress_text = st.sidebar.empty()
                for i, scheme_str in enumerate(selected_schemes):
                    code = scheme_str.split(" - ")[0]
                    name = scheme_str.split(" - ")[1]
                    progress_text.text(f"Fetching: {name[:20]}...")
                    df_fund = fetch_amfi_nav(code)
                    if not df_fund.empty:
                        dfs.append(df_fund)
                progress_text.empty()
                if dfs:
                    merged_df = pd.concat(dfs, axis=1)
                    merged_df.sort_index(inplace=True)
                    st.session_state['amfi_data'] = merged_df
                    st.success(f"✅ Loaded {len(dfs)} funds!")
                else:
                    st.error("Could not fetch data.")
            else:
                st.warning("Please select at least one fund.")
    
    if not st.session_state['amfi_data'].empty:
        df_global = st.session_state['amfi_data']

sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", value=10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Simulation Settings")
num_simulations = st.sidebar.number_input("Simulations", 500, 50000, 3000, step=500)

with st.sidebar.expander("🔎 Advanced Filters", expanded=False):
    filter_romad = st.slider("RoMaD Range", 0.0, 5.0, (0.0, 5.0), step=0.1)
    filter_return = st.slider("Annual Return %", 0.0, 100.0, (0.0, 100.0), step=1.0)
    filter_rolling = st.slider("Avg 3Y Rolling %", 0.0, 100.0, (0.0, 100.0), step=1.0)
    filter_sharpe = st.slider("Sharpe Ratio", 0.0, 5.0, (0.0, 5.0), step=0.1)
    filter_div = st.slider("Diversification Score", 1.0, 3.0, (1.0, 3.0), step=0.1)

# --- MAIN APP LOGIC ---
if not df_global.empty:
    all_funds = df_global.columns.tolist()
    if data_source == "Google Sheet":
        st.subheader("1. Select Funds")
        selected_funds = st.multiselect("Select Funds (Min 2)", all_funds, default=all_funds[:2])
    else:
        st.subheader("1. Selected Funds")
        selected_funds = st.multiselect("Funds to Analyze", all_funds, default=all_funds)

    c1, c2 = st.columns([1, 1])
    
    if len(selected_funds) > 1:
        df_selected = df_global[selected_funds]
        valid_data = df_selected.dropna()
        
        if valid_data.empty:
            st.error("❌ No overlapping dates found.")
        else:
            min_date = valid_data.index.min().date()
            max_date = valid_data.index.max().date()
            
            with c2:
                st.write("### 📅 Select Time Period")
                col_start, col_end = st.columns(2)
                start_date = col_start.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
                end_date = col_end.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
                
                if start_date >= end_date:
                    st.error("Start Date must be before End Date.")
                    df_filtered = pd.DataFrame()
                else:
                    df_filtered = valid_data.loc[str(start_date):str(end_date)]
                    st.success(f"Analyzing {len(df_filtered)} trading days.")

            if not df_filtered.empty:
                tab1, tab2, tab3 = st.tabs(["🚀 Strategy Lab", "🧩 Correlation Matrix", "⚖️ Fund Comparison"])

                # ==========================
                # TAB 1: STRATEGY LAB
                # ==========================
                with tab1:
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
                            mc4.metric("Div. Score", f"{m_metrics['DivRatio']:.2f}")
                            fig = px.line(m_df, x='Date', y=['Invested', 'Value'], 
                                            color_discrete_map={'Invested':'#D3D3D3', 'Value':'#3b82f6'})
                            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)

                    st.markdown("---")
                    st.subheader("3. AI Optimization Engine")
                    opt_col1, opt_col2 = st.columns([1, 3])
                    with opt_col1:
                        optimize_for = st.selectbox(
                            "Maximize For:",
                            ("Diversification (Min Correlation)", "Rolling Returns (Consistency)", "RoMaD (Max Safety)", "Sharpe (Max Efficiency)", "Returns (Max Profit)")
                        )
                        use_best_x = st.checkbox("Pick Subset?", value=False)
                        if use_best_x:
                            x_funds = st.slider("Number of funds (X)", 2, len(selected_funds), 2)
                        else:
                            x_funds = len(selected_funds)
                        run_opt = st.button("🚀 Run AI Optimizer")

                    if run_opt:
                        with st.spinner(f"Simulating {num_simulations} portfolios..."):
                            daily_returns = df_filtered.pct_change().dropna()
                            mean_returns = daily_returns.mean() * 252 
                            cov_matrix = daily_returns.cov() * 252
                            individual_vols = daily_returns.std() * np.sqrt(252)
                            
                            sim_data = []
                            roll_window = 756
                            
                            for i in range(num_simulations):
                                w = np.random.random(len(selected_funds))
                                if use_best_x:
                                    top_indices = np.argpartition(w, -x_funds)[-x_funds:]
                                    mask = np.zeros_like(w)
                                    mask[top_indices] = w[top_indices]
                                    w = mask
                                w /= np.sum(w)
                                
                                ret = np.sum(mean_returns * w)
                                port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                                sharpe = ret / port_vol if port_vol > 0 else 0
                                weighted_vol = np.sum(w * individual_vols)
                                div_ratio = weighted_vol / port_vol if port_vol > 0 else 0
                                
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
                                    "Sharpe": sharpe,
                                    "DivRatio": div_ratio
                                })
                            
                            sim_df = pd.DataFrame(sim_data)
                            mask = (
                                (sim_df["RoMaD"].between(filter_romad[0], filter_romad[1])) &
                                (sim_df["Returns"].between(filter_return[0], filter_return[1])) &
                                (sim_df["Rolling"].between(filter_rolling[0], filter_rolling[1])) &
                                (sim_df["Sharpe"].between(filter_sharpe[0], filter_sharpe[1])) &
                                (sim_df["DivRatio"].between(filter_div[0], filter_div[1]))
                            )
                            filtered_df = sim_df[mask]
                            
                            if filtered_df.empty:
                                st.warning("⚠️ No match found. Showing best Unfiltered result.")
                                final_df = sim_df
                            else:
                                st.success(f"✅ Found {len(filtered_df)} matching portfolios.")
                                final_df = filtered_df
                            
                            if "RoMaD" in optimize_for: best_row = final_df.loc[final_df['RoMaD'].idxmax()]
                            elif "Sharpe" in optimize_for: best_row = final_df.loc[final_df['Sharpe'].idxmax()]
                            elif "Rolling" in optimize_for: best_row = final_df.loc[final_df['Rolling'].idxmax()]
                            elif "Diversification" in optimize_for: best_row = final_df.loc[final_df['DivRatio'].idxmax()]
                            else: best_row = final_df.loc[final_df['Returns'].idxmax()]
                            
                            best_weights = {f: w*100 for f, w in zip(selected_funds, best_row['weights'])}
                            ai_metrics, ai_df = run_backtest(df_filtered, best_weights, sip_amount)
                            
                            comp_col1, comp_col2 = st.columns(2)
                            with comp_col1:
                                st.write("**Manual**")
                                man_df = pd.DataFrame.from_dict(manual_weights, orient='index', columns=['Weight'])
                                st.dataframe(man_df.style.format("{:.1f}%"))
                            with comp_col2:
                                st.write(f"**AI Strategy ({optimize_for})**")
                                ai_alloc_df = pd.DataFrame.from_dict(best_weights, orient='index', columns=['Weight'])
                                st.dataframe(ai_alloc_df.style.format("{:.1f}%"))
                            
                            comp_metrics = pd.DataFrame({
                                "Metric": ["XIRR", "Div Score", "RoMaD", "Max DD", "Sharpe", "Final Value"],
                                "Your Strategy": [
                                    f"{m_metrics['XIRR']:.2f}%", f"{m_metrics['DivRatio']:.2f}", f"{m_metrics['RoMaD']:.2f}", 
                                    f"{m_metrics['MaxDD']:.2f}%", f"{m_metrics['Sharpe']:.2f}", f"₹{m_metrics['Current Value']:,.0f}"
                                ],
                                "AI Strategy": [
                                    f"{ai_metrics['XIRR']:.2f}%", f"{ai_metrics['DivRatio']:.2f}", f"{ai_metrics['RoMaD']:.2f}", 
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

                # ==========================
                # TAB 2: CORRELATION MATRIX
                # ==========================
                with tab2:
                    st.subheader("🧩 Fund Correlation Matrix")
                    st.caption("Lower numbers (Blue) = Better Diversification. Red = Duplication.")
                    daily_ret = df_filtered.pct_change().dropna()
                    corr_matrix = daily_ret.corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    st.subheader("🔄 Rolling Correlation (Dynamic)")
                    f1 = st.selectbox("Fund A", selected_funds, index=0)
                    f2 = st.selectbox("Fund B", selected_funds, index=1 if len(selected_funds)>1 else 0)
                    if f1 != f2:
                        roll_corr = daily_ret[f1].rolling(126).corr(daily_ret[f2]) 
                        fig_roll = px.line(roll_corr, title=f"6-Month Rolling Correlation: {f1} vs {f2}")
                        fig_roll.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Uncorrelated")
                        fig_roll.update_yaxes(range=[-1, 1])
                        st.plotly_chart(fig_roll, use_container_width=True)
                    else:
                        st.info("Select two different funds to see their dynamic relationship.")

                # ==========================
                # TAB 3: FUND COMPARISON
                # ==========================
                with tab3:
                    st.subheader("⚖️ Head-to-Head Fund Comparison")
                    st.caption("Analyzing funds individually to spot winners and losers.")

                    comp_data = []
                    daily_ret_comp = df_filtered.pct_change().dropna()
                    # Calculate Correlation of each fund to the 'Average of Others'
                    corr_matrix = daily_ret_comp.corr()
                    
                    for fund in selected_funds:
                        # 1. SIP XIRR
                        single_fund_metrics, _ = run_backtest(df_filtered[[fund]], {fund: 100}, sip_amount)
                        sip_xirr = single_fund_metrics['XIRR'] if single_fund_metrics else 0
                        
                        # 2. Lump Sum CAGR & Absolute
                        start_price = df_filtered[fund].iloc[0]
                        end_price = df_filtered[fund].iloc[-1]
                        abs_ret = ((end_price - start_price) / start_price) * 100
                        days = (df_filtered.index[-1] - df_filtered.index[0]).days
                        cagr = ((end_price / start_price) ** (365.25/days) - 1) * 100 if days > 0 else 0
                        
                        # 3. Risk Metrics
                        fund_daily = daily_ret_comp[fund]
                        volatility = fund_daily.std() * np.sqrt(252) * 100
                        
                        # Diversifier Score (1 - Avg Correlation to others)
                        other_funds = [f for f in selected_funds if f != fund]
                        if other_funds:
                            avg_corr = corr_matrix[fund][other_funds].mean()
                            div_score = (1 - avg_corr) * 10  # Scale 0-10
                        else:
                            div_score = 0
                            
                        # Max Drawdown
                        cum_ret = (1 + fund_daily).cumprod()
                        dd = (cum_ret - cum_ret.cummax()) / cum_ret.cummax()
                        max_dd_val = abs(dd.min()) * 100

                        comp_data.append({
                            "Fund Name": fund,
                            "SIP XIRR": f"{sip_xirr:.2f}%",
                            "Lump Sum CAGR": f"{cagr:.2f}%",
                            "Volatility (Risk)": f"{volatility:.2f}%",
                            "Max Drawdown": f"{max_dd_val:.2f}%",
                            "Diversifier Score (0-10)": f"{div_score:.1f}"
                        })

                    st.dataframe(pd.DataFrame(comp_data).set_index("Fund Name"), use_container_width=True)
                    st.info("💡 **Diversifier Score:** Higher is better. It means this fund behaves differently from the rest of your selection.")

                    col_g, col_dd = st.columns(2)
                    
                    with col_g:
                        st.subheader("📈 Normalized Growth (Base 100)")
                        # Rebase all funds to start at 100
                        normalized_df = df_filtered / df_filtered.iloc[0] * 100
                        fig_norm = px.line(normalized_df, title="Hypothetical ₹100 invested in each fund")
                        st.plotly_chart(fig_norm, use_container_width=True)

                    with col_dd:
                        st.subheader("📉 Drawdown Comparison")
                        # Calculate Drawdown for all funds
                        dd_df = pd.DataFrame()
                        for fund in selected_funds:
                            cum = (1 + daily_ret_comp[fund]).cumprod()
                            dd_df[fund] = (cum - cum.cummax()) / cum.cummax()
                        
                        fig_dd_all = px.area(dd_df, title="Crash Depth over Time")
                        fig_dd_all.update_yaxes(tickformat=".1%")
                        st.plotly_chart(fig_dd_all, use_container_width=True)

    else:
        st.info("👈 Select at least 2 funds.")
else:
    st.info("👈 Choose a Data Source to begin.")
