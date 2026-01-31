import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import io

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

@st.cache_data
def load_data(sheet_url):
    """Robust data loader that handles commas, quotes, and empty cells"""
    try:
        csv_url = get_csv_url(sheet_url)
        
        # Fetch Data
        response = requests.get(csv_url)
        response.raise_for_status()
        
        if "<!DOCTYPE html>" in response.text[:100]:
            st.error("⚠️ Error: The Google Sheet is not public. Please change permissions to 'Anyone with the link'.")
            return pd.DataFrame()

        # Parse CSV
        df = pd.read_csv(io.StringIO(response.text), on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        # Handle Date Column
        date_col = next((col for col in df.columns if col.lower() == 'date'), None)
        if not date_col:
            st.error("⚠️ Error: Could not find a 'Date' column.")
            return pd.DataFrame()
            
        df.rename(columns={date_col: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # Drop rows ONLY if the Date itself is missing
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        
        # --- ROBUST NUMBER CLEANING ---
        # 1. Convert all columns to string first
        # 2. Remove commas, quotes, and spaces
        # 3. Convert back to numeric
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # CRITICAL CHANGE: Do NOT dropna() here. 
        # We wait until the user selects funds.
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
        # Fund Selection
        all_funds = df.columns.tolist()
        st.sidebar.markdown("---")
        selected_funds = st.sidebar.multiselect("Select Funds to Analyze", all_funds, default=all_funds[:2])
        
        if selected_funds:
            # --- SMART FILTERING ---
            # 1. Filter by Time
            end_date = df.index.max()
            start_date = end_date - pd.DateOffset(years=years)
            
            # 2. Slice Data (Select only the requested columns and rows)
            df_filtered = df.loc[start_date:end_date, selected_funds]
            
            # 3. NOW we drop rows where THESE SPECIFIC funds are missing data
            # "how='any'" means if any of the selected funds is missing a price today, skip the day.
            df_filtered = df_filtered.dropna(how='any')
            
            if df_filtered.empty:
                st.warning("⚠️ No common data found for these funds in this time range. They might have different start dates.")
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
                
                # Show allocation
                cols = st.columns(len(selected_funds))
                for i, (f, w) in enumerate(best_allocation.items()):
                    cols[i].metric(f, f"{w*100:.1f}%")

                # --- 2. BACKTEST ---
                st.subheader("2. Backtest Results")
                monthly_data = df_filtered.resample('MS').first()
                total_invested = 0
                units = {f:0.0 for f in selected_funds}
                ledger = []
                
                for date, row in monthly_data.iterrows():
                    total_invested += sip_amount
                    for f in selected_funds:
                        units[f] += (sip_amount * best_allocation[f]) / row[f]
                    
                    curr_val = sum([units[f] * row[f] for f in selected_funds])
                    ledger.append({'Date': date, 'Invested': total_invested, 'Value': curr_val})
                
                res_df = pd.DataFrame(ledger)
                
                if not res_df.empty:
                    final_val = res_df.iloc[-1]['Value']
                    invested = res_df.iloc[-1]['Invested']
                    profit = final_val - invested
                    
                    # Calculate XIRR/CAGR approx
                    years_actual = (res_df.iloc[-1]['Date'] - res_df.iloc[0]['Date']).days / 365.25
                    cagr = ((final_val / invested)**(1/years_actual) - 1) * 100 if years_actual > 0 else 0

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Invested", f"₹{invested:,.0f}")
                    c2.metric("Final Value", f"₹{final_val:,.0f}")
                    c3.metric("Net Profit", f"₹{profit:,.0f}")
                    c4.metric("CAGR (Approx)", f"{cagr:.1f}%")
                    
                    fig = px.line(res_df, x='Date', y=['Invested', 'Value'], 
                                  color_discrete_map={'Invested':'#D3D3D3', 'Value':'#00CC96'})
                    fig.update_traces(fill='tozeroy')
                    st.plotly_chart(fig, use_container_width=True)
                    
        else:
            st.info("👈 Please select funds from the sidebar.")
