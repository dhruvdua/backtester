import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SIP Backtest Pro", layout="wide")
st.title("💰 Mutual Fund SIP Backtester & Optimizer")
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def get_csv_url(url):
    """
    Converts a standard Google Sheet 'edit' URL into a downloadable CSV URL.
    """
    if "docs.google.com/spreadsheets" not in url:
        return url  # Return as is if it's likely a direct CSV link
    
    # 1. Base URL cleanup (remove /edit and everything after)
    base_url = url.split('/edit')[0]
    
    # 2. Check for 'gid' (Sheet ID) to grab the specific tab
    gid = "0" # Default to first sheet
    if "gid=" in url:
        import re
        gid_match = re.search(r'gid=(\d+)', url)
        if gid_match:
            gid = gid_match.group(1)
            
    # 3. Construct the export URL
    return f"{base_url}/export?format=csv&gid={gid}"

@st.cache_data
def load_data(sheet_url):
    """
    Robust data loading function that handles Google Login pages,
    bad rows, and text formatting issues.
    """
    try:
        csv_url = get_csv_url(sheet_url)
        
        # Read CSV, skipping bad lines
        df = pd.read_csv(csv_url, on_bad_lines='skip')
        
        # Check if we got a Google Login page HTML instead of data
        if df.columns[0].strip().startswith('<'):
            st.error("⚠️ Error: The code downloaded a Login Page. Please set your Google Sheet access to 'Anyone with the link'.")
            return pd.DataFrame()

        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Ensure 'Date' column exists (case-insensitive check)
        date_col = next((col for col in df.columns if col.lower() == 'date'), None)
        if not date_col:
            st.error("⚠️ Error: Could not find a column named 'Date' in your sheet.")
            return pd.DataFrame()
            
        # Rename valid date column to 'Date' for consistency
        df.rename(columns={date_col: 'Date'}, inplace=True)

        # Force Date parsing (Coerce errors to NaT)
        # Try parsing with dayfirst=True (handles DD-MM-YYYY which is common in India)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Set Index
        df.set_index('Date', inplace=True)
        
        # Force numeric conversion for all other columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.dropna() # Final cleanup
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return pd.DataFrame()

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")

# Default URL is your example sheet
default_url = "https://docs.google.com/spreadsheets/d/1lMNqoG0Z_WehJVUlP4-3Jf6cDofjUOu5FMK-xJ1EnOU/edit?gid=0#gid=0"
sheet_url = st.sidebar.text_input("Paste Google Sheet URL", value=default_url)

sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", value=10000, step=1000)
years = st.sidebar.slider("Analysis Duration (Years)", min_value=1, max_value=10, value=3)

# --- MAIN APP LOGIC ---

if sheet_url:
    with st.spinner("Fetching and processing data..."):
        df = load_data(sheet_url)
    
    if not df.empty:
        # Fund Selection
        all_funds = df.columns.tolist()
        st.sidebar.markdown("---")
        selected_funds = st.sidebar.multiselect("Select Funds to Analyze", all_funds, default=all_funds[:2])
        
        if selected_funds:
            # Filter Data by Time (Backdate from today)
            end_date = df.index.max()
            start_date = end_date - pd.DateOffset(years=years)
            
            # Slice the dataframe
            df_filtered = df.loc[start_date:end_date, selected_funds].dropna()
            
            if df_filtered.empty:
                st.warning("No data found for the selected date range. Try reducing the number of years.")
            else:
                st.write(f"Analyzing data from **{start_date.date()}** to **{end_date.date()}** ({len(df_filtered)} trading days)")

                # --- 1. MONTE CARLO OPTIMIZATION ---
                st.markdown("### 1. 🧠 AI Optimized Allocation")
                st.info("We simulated 2,000 portfolio combinations to find the best risk-reward ratio (Sharpe Ratio).")
                
                # Calculate Daily Returns
                returns = df_filtered.pct_change()
                mean_returns = returns.mean() * 252 # Annualized
                cov_matrix = returns.cov() * 252    # Annualized
                
                # Simulation Arrays
                num_portfolios = 2000
                results_arr = np.zeros((3, num_portfolios))
                all_weights = []
                
                # Run Simulation
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

                # --- 2. SIP BACKTEST ---
                st.markdown("### 2. 📈 SIP Performance Backtest")
                
                # Resample to Monthly for SIP (Buying on 1st of month)
                monthly_data = df_filtered.resample('MS').first()
                
                # Initialize tracking
                total_invested = 0
                units_held = {fund: 0.0 for fund in selected_funds}
                ledger = []

                for date, row in monthly_data.iterrows():
                    total_invested += sip_amount
                    
                    # Buy units
                    for fund in selected_funds:
                        allocation_amt = sip_amount * best_allocation[fund]
                        price = row[fund]
                        if price > 0: 
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
                    
                    # CAGR Approximation
                    years_actual = (res_df.iloc[-1]['Date'] - res_df.iloc[0]['Date']).days / 365.25
                    cagr = ((final_val / final_inv)**(1/years_actual) - 1) * 100 if years_actual > 0 else 0

                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Invested", f"₹{final_inv:,.0f}")
                    m2.metric("Final Value", f"₹{final_val:,.0f}")
                    m3.metric("Net Profit", f"₹{abs_return:,.0f}")
                    m4.metric("CAGR (Approx)", f"{cagr:.2f}%")

                    # Visuals
                    fig = px.line(res_df, x='Date', y=['Invested', 'Portfolio Value'], 
                                  title="Portfolio Growth vs Investment",
                                  color_discrete_map={"Invested": "#FFA500", "Portfolio Value": "#00CC96"})
                    
                    # Add Area fill
                    fig.update_traces(fill='tozeroy')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show Raw Data (optional expander)
                    with st.expander("See Monthly Breakdown"):
                        st.dataframe(res_df.style.format({"Invested": "₹{:.2f}", "Portfolio Value": "₹{:.2f}"}))

        else:
            st.info("👈 Please select at least one fund from the sidebar to begin.")
