import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="SIP Backtest Pro", layout="wide")
st.title("💰 Mutual Fund SIP Backtester & Optimizer")

# --- Helper Function: Convert Google Sheet URL to CSV ---
def get_csv_url(url):
    """
    Converts a standard Google Sheet 'edit' URL into a downloadable CSV URL.
    Handles specific sheet IDs (gid) if present.
    """
    if "docs.google.com/spreadsheets" not in url:
        return url  # Return as is if it's not a google sheet (maybe it's a direct csv link)
    
    # 1. Base URL cleanup
    base_url = url.split('/edit')[0]
    
    # 2. Check for 'gid' (Sheet ID) to grab the specific tab
    # The gid is usually in the params or hash like #gid=12345
    gid = "0" # Default to first sheet
    if "gid=" in url:
        # Extract gid from URL parameters
        import re
        gid_match = re.search(r'gid=(\d+)', url)
        if gid_match:
            gid = gid_match.group(1)
            
    # 3. Construct the export URL
    final_url = f"{base_url}/export?format=csv&gid={gid}"
    return final_url

# --- 1. Data Loading ---
# --- 1. Data Loading (Robust Version) ---
@st.cache_data
def load_data(sheet_url):
    try:
        csv_url = get_csv_url(sheet_url)
        
        # 1. Read the CSV
        # on_bad_lines='skip' helps if some rows have too many/few commas
        df = pd.read_csv(csv_url, on_bad_lines='skip')
        
        # 2. Check if we actually got data or a Google Login page
        if df.columns[0] == '<!DOCTYPE html>':
            st.error("Error: The code downloaded a Login Page instead of data. Please set your Google Sheet access to 'Anyone with the link'.")
            return pd.DataFrame()

        # 3. Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # 4. Flexible Date Parsing
        # errors='coerce' turns non-dates (like "Total" or empty rows) into NaT (Not a Time)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # 5. Drop rows where Date is invalid (NaT)
        df = df.dropna(subset=['Date'])
        
        # 6. Set Index
        df.set_index('Date', inplace=True)
        
        # 7. Ensure all other columns are numeric
        # This converts string numbers like "1,000" to 1000 and "Text" to NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.dropna() # Final cleanup of any rows with bad number data

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# --- 2. Sidebar Inputs ---
st.sidebar.header("Configuration")
default_url = "https://docs.google.com/spreadsheets/d/1lMNqoG0Z_WehJVUlP4-3Jf6cDofjUOu5FMK-xJ1EnOU/edit?gid=0#gid=0"
sheet_url = st.sidebar.text_input("Paste Google Sheet URL", value=default_url)
sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", value=10000, step=1000)
years = st.sidebar.slider("Investment Duration (Years)", min_value=1, max_value=10, value=3)

# --- Main Logic ---
if sheet_url:
    df = load_data(sheet_url)
    
    if not df.empty:
        # Fund Selection
        all_funds = df.columns.tolist()
        selected_funds = st.sidebar.multiselect("Select Funds to Analyze", all_funds, default=all_funds[:2])
        
        if selected_funds:
            # Filter Data by Time (Backdate from today)
            end_date = df.index.max()
            start_date = end_date - pd.DateOffset(years=years)
            
            # Slice the dataframe to the relevant window and funds
            # We use 'loc' to slice by date index
            df_filtered = df.loc[start_date:end_date, selected_funds].dropna()
            
            st.write(f"Analyzing data from **{start_date.date()}** to **{end_date.date()}**")

            # --- 3. Monte Carlo Optimization ---
            st.subheader("1. AI Optimized Strategy")
            
            # Calculate Daily Returns
            returns = df_filtered.pct_change()
            mean_returns = returns.mean() * 252
            cov_matrix = returns.cov() * 252
            
            # Simulation
            num_portfolios = 2000
            results_arr = np.zeros((3, num_portfolios))
            all_weights = []
            
            for i in range(num_portfolios):
                weights = np.random.random(len(selected_funds))
                weights /= np.sum(weights)
                all_weights.append(weights)
                
                p_ret = np.sum(mean_returns * weights)
                p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results_arr[0,i] = p_ret
                results_arr[1,i] = p_vol
                results_arr[2,i] = p_ret / p_vol # Sharpe Ratio

            # Get Best Portfolio
            max_sharpe_idx = np.argmax(results_arr[2])
            best_weights = all_weights[max_sharpe_idx]
            best_allocation = dict(zip(selected_funds, best_weights))

            # Display Optimal Allocation
            cols = st.columns(len(selected_funds))
            for idx, (fund, weight) in enumerate(best_allocation.items()):
                cols[idx].metric(label=fund, value=f"{weight*100:.1f}%")

            # --- 4. SIP Backtest Implementation ---
            st.subheader("2. SIP Performance Check")
            
            # Resample to Monthly for SIP (Buying on 1st of month)
            monthly_data = df_filtered.resample('MS').first()
            
            # Initialize tracking
            total_invested = 0
            # Current value starts at 0
            ledger = []

            # We need to track units accumulated per fund
            units_held = {fund: 0.0 for fund in selected_funds}

            for date, row in monthly_data.iterrows():
                total_invested += sip_amount
                
                # Buy units
                for fund in selected_funds:
                    allocation_amt = sip_amount * best_allocation[fund]
                    price = row[fund]
                    if price > 0: # Avoid division by zero
                        units = allocation_amt / price
                        units_held[fund] += units
                
                # Calculate Portfolio Value on this date
                current_value = sum([units_held[f] * row[f] for f in selected_funds])
                
                ledger.append({
                    'Date': date,
                    'Invested': total_invested,
                    'Portfolio Value': current_value
                })
            
            # Create DataFrame for results
            res_df = pd.DataFrame(ledger)
            
            if not res_df.empty:
                final_val = res_df.iloc[-1]['Portfolio Value']
                final_inv = res_df.iloc[-1]['Invested']
                abs_return = final_val - final_inv
                xirr_approx = ((final_val / final_inv)**(1/years) - 1) * 100

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Invested", f"₹{final_inv:,.0f}")
                m2.metric("Final Value", f"₹{final_val:,.0f}")
                m3.metric("Net Profit", f"₹{abs_return:,.0f}", delta=f"{xirr_approx:.2f}% CAGR (Approx)")

                # Visuals
                fig = px.line(res_df, x='Date', y=['Invested', 'Portfolio Value'], 
                              color_discrete_map={"Invested": "gray", "Portfolio Value": "#00CC96"})
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Please select at least one fund from the sidebar.")
