import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(page_title="SIP Backtester", layout="wide")
st.title("💰 Mutual Fund SIP Backtester & Optimizer")

@st.cache_data
def load_data(sheet_url):
    # Read CSV from the published Google Sheet URL
    df = pd.read_csv(sheet_url)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Sidebar Inputs
sheet_url = st.sidebar.text_input("Paste Google Sheet CSV URL")
sip_amount = st.sidebar.number_input("Monthly SIP Amount (₹)", value=10000)
years = st.sidebar.number_input("Duration (Years)", value=5)

if sheet_url:
    data = load_data(sheet_url)
    fund_list = data.columns.tolist()
    selected_funds = st.sidebar.multiselect("Select Funds", fund_list, default=fund_list[:2])
    
    # Filter data for the selected duration (Backdate from today)
    start_date = data.index.max() - pd.DateOffset(years=years)
    df_filtered = data.loc[start_date:data.index.max(), selected_funds]

def optimize_portfolio(prices_df):
    # Calculate daily returns
    returns = prices_df.pct_change()
    mean_returns = returns.mean() * 252 # Annualized
    cov_matrix = returns.cov() * 252

    num_portfolios = 5000 # Monte Carlo iterations
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        # Generate random weights
        weights = np.random.random(len(prices_df.columns))
        weights /= np.sum(weights) # Normalize to equal 1 (100%)
        weights_record.append(weights)

        # Calculate Portfolio Return & Volatility
        p_return = np.sum(mean_returns * weights)
        p_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Store results (Return, Risk, Sharpe Ratio)
        results[0,i] = p_return
        results[1,i] = p_std_dev
        results[2,i] = p_return / p_std_dev # Sharpe Ratio

    # Find the portfolio with Max Sharpe Ratio
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]
    
    return optimal_weights, results

if sheet_url and selected_funds:
    st.subheader("🤖 Calculating Optimal Allocation...")
    opt_weights, sim_results = optimize_portfolio(df_filtered)
    
    # Display Optimal Weights
    best_allocation = dict(zip(selected_funds, opt_weights))
    st.write("Based on Monte Carlo simulations, the ideal allocation to maximize risk-adjusted returns is:")
    st.json({k: f"{v*100:.2f}%" for k, v in best_allocation.items()})

def calculate_sip(df, allocations, monthly_amt):
    # Resample data to Monthly (taking the first price of the month)
    monthly_data = df.resample('MS').first()
    
    total_invested = 0
    portfolio_units = {fund: 0 for fund in allocations.keys()}
    portfolio_value_over_time = []

    for date, row in monthly_data.iterrows():
        total_invested += monthly_amt
        
        # Buy units for each fund based on allocation
        for fund, weight in allocations.items():
            fund_amt = monthly_amt * weight
            units_bought = fund_amt / row[fund]
            portfolio_units[fund] += units_bought
        
        # Calculate current value of portfolio
        current_val = 0
        for fund in allocations.keys():
            current_val += portfolio_units[fund] * row[fund]
        
        portfolio_value_over_time.append({'Date': date, 'Portfolio Value': current_val, 'Invested': total_invested})

    return pd.DataFrame(portfolio_value_over_time)

if sheet_url and selected_funds:
    # Run SIP calculation using the Optimal Weights found above
    backtest_df = calculate_sip(df_filtered, best_allocation, sip_amount)
    
    final_value = backtest_df.iloc[-1]['Portfolio Value']
    total_investment = backtest_df.iloc[-1]['Invested']
    
    st.metric("Total Invested", f"₹{total_investment:,.2f}")
    st.metric("Current Value", f"₹{final_value:,.2f}", delta=f"{(final_value-total_investment)/total_investment*100:.1f}% Return")
    
    # Chart
    fig = px.line(backtest_df, x='Date', y=['Portfolio Value', 'Invested'], title="SIP Performance Curve")
    st.plotly_chart(fig)

