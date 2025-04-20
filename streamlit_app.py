import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.tsa.stattools as ts
import numpy as np
from pykalman import KalmanFilter

def get_data(tickers, start_d, end_d):
    data = pd.DataFrame()
    names = list()

    for i in tickers:
        data = pd.concat([data, pd.DataFrame(yf.download(i, start=start_d, end=end_d).iloc[:,0])], axis=1)
        names.append(i)

    data.columns = names
    
    return data

def get_pair():
    ticks = [first_stock_input, second_stock_input]
    final_data = get_data(ticks, start_input, end_input)
    first_stock = final_data[first_stock_input]
    second_stock = final_data[second_stock_input]
    model_choice = model_input

    return [final_data, first_stock, second_stock, model_choice]

def stat_tests():
    d_list = get_pair()
    first_stock = d_list[1]
    second_stock = d_list[2]

    coint_result = ts.coint(first_stock, second_stock)
    coint_t_statistic = coint_result[0]
    coint_p_value = coint_result[1]
    crit_value_statistic = coint_result[2]

    first_stock_ADF = ts.adfuller(first_stock)
    second_stock_ADF = ts.adfuller(second_stock)
    spread_ADF = ts.adfuller(first_stock - second_stock)
    ratio_ADF = ts.adfuller(first_stock / second_stock)

    return [coint_p_value, first_stock_ADF[1], second_stock_ADF[1], spread_ADF[1], ratio_ADF[1]]

def pair_analysis():
    d_list = get_pair()
    final_data = d_list[0]
    first_stock = d_list[1]
    second_stock = d_list[2]
    model_choice = d_list[3]

    stock_pair_fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(first_stock)
    plt.plot(second_stock)
    plt.title("Pricing Comparison")
    plt.legend([first_stock_input, second_stock_input])
    st.pyplot(stock_pair_fig)

    #display spread
    spread_pair_fig, ax = plt.subplots(figsize=(8,6))
    spread_stocks = first_stock - second_stock
    plt.plot(spread_stocks)
    plt.axhline(spread_stocks.mean(), color="red")
    plt.legend(["Spread"])
    plt.title("Price Spread")
    st.pyplot(spread_pair_fig)

    #display ratio
    ratio_pair_fig, ax = plt.subplots(figsize=(8,6))
    ratio_stocks = (first_stock / second_stock)
    plt.plot(ratio_stocks)
    plt.axhline(ratio_stocks.mean(), color="red")
    plt.legend(["Ratio"])
    plt.title("Ratio of stocks")
    st.pyplot(ratio_pair_fig)

    if model_choice == ("Kalman Filter"):
        z_score_graph = (spread_stocks - spread_stocks.mean()) / spread_stocks.std()
        type_graph = "Spread"

        #Kalman Filter
        x_matrix = np.array([[[x_i, 1.0]] for x_i in first_stock])

        kf = KalmanFilter(
            transition_matrices = np.eye(2),
            observation_matrices = x_matrix,
            initial_state_mean = [0, 0],
            observation_covariance = 0.1,
            transition_covariance = 0.05 * np.eye(2)
        )

        state_means, _ = kf.filter(second_stock)
        beta, intercept = state_means.T

        spread = second_stock - (beta * first_stock + intercept)

        window = 90  
        spread_series = pd.Series(spread)
        spread_mean = spread_series.rolling(window).mean()
        spread_std = spread_series.rolling(window).std()
        zscore = (spread_series - spread_mean) / spread_std

        short_second = pd.Series(index=zscore.index, data=np.nan)
        long_second = pd.Series(index=zscore.index, data=np.nan)

        short_cross = (zscore < -1)
        long_cross = (zscore > 1)
        short_second[short_cross] = spread_mean[short_cross]
        long_second[long_cross] = spread_mean[long_cross]
    else:
        z_score_graph = (ratio_stocks - ratio_stocks.mean()) / ratio_stocks.std()
        type_graph = "Ratio"

        ratio_ma_5 = ratio_stocks.rolling(window=5, center=False).mean()
        ratio_ma_14 = ratio_stocks.rolling(window=14, center=False).mean()
        std_14 = ratio_stocks.rolling(window=14, center=False).std()
        z_score_20_5 = (ratio_ma_5 - ratio_ma_14) / std_14
        
        #moving averages of 5 and 14 days
        moving_average_fig, ax = plt.subplots(figsize=(8,6))
        plt.plot(ratio_stocks.index, ratio_stocks.values)
        plt.plot(ratio_ma_5.index, ratio_ma_5.values)
        plt.plot(ratio_ma_14.index, ratio_ma_14.values)
        plt.legend(["Ratio", "5 day MA", "14 day MA"])
        plt.title("Moving Averages for 5 and 14 days")
        st.pyplot(moving_average_fig)

        buy = ratio_stocks.copy()
        sell = ratio_stocks.copy()
        buy[z_score_20_5 > -1] = 0
        sell[z_score_20_5 < 1] = 0

    #Rolling Z Score of Spread/Ratio
    rolling_zscore_fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(z_score_graph)
    plt.axhline(0, color="black")
    plt.axhline(1, color="red", linestyle="--")
    plt.axhline(1.25, color="red", linestyle="--")
    plt.axhline(-1, color="green", linestyle="--")
    plt.axhline(-1.25, color="green", linestyle="--")
    plt.legend(["Rolling Z Score"])
    plt.title(f"Z score bands on {type_graph}")
    st.pyplot(rolling_zscore_fig)

    #Rolling Z score with trading signals
    signals_fig, ax = plt.subplots(figsize=(8,6))
    
    if model_choice == ("Kalman Filter"):
        plt.plot(spread_mean)
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, spread_mean.min(), spread_mean.max()))
        plt.plot(short_second, color="green", linestyle="None", marker="^")
        plt.plot(long_second, color="red", linestyle="None", marker="^")
    else:
        plt.plot(ratio_stocks)
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, ratio_stocks.min(), ratio_stocks.max()))
        plt.plot(buy, color="green", linestyle="None", marker="^")
        plt.plot(sell, color="red", linestyle="None", marker="^")
        
    plt.legend([f"{type_graph}", "Long B/Short A", "Short B/Long A"])
    plt.title(f"Rolling Z Score of {type_graph}")
    st.pyplot(signals_fig)

#main
st.set_page_config(layout="wide")
middle, right = st.columns([0.675, 0.325])

st.sidebar.title("Mean Reversion Pair Analysis")
st.sidebar.markdown("""
        <p style="
            background-color: #1D1B1B;
            padding: 1px 5px;
            border-radius: 8px;
            font-family: Gill Sans;
            font-size: 14px;
            color: #009933;
            width: fit-content;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 0px;
        "> 
            Created by:
        </p>
        <div style="
            display: flex;
            align-items: center;
            padding: 8px 10px;
            width: fit-content;
        ">
            <a href="https://linkedin.com/in/edwardhuangg">
                <i class="ion-social-linkedin" style="font-size: 30px; margin-right: 8px;"></i>
            </a>
            <h6 style="
                    background-color: #1D1B1B;
                    padding: 3px 5px;
                    border-radius: 8px;
                    font-family: Gill Sans;
                    font-size: 14px;
                    color: white;
                    width: fit-content;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            ">
                Edward Huang
            </h6>
        </div>

        <link href="https://code.ionicframework.com/ionicons/2.0.1/css/ionicons.min.css" rel="stylesheet">
    """, unsafe_allow_html=True)

with st.sidebar.form("Choose two stocks"):
    first_stock_input = st.text_input("stock A", value="NKE")
    second_stock_input = st.text_input("stock B", value="AAPl")
    start_input = st.date_input(label = "From: ", format="YYYY/MM/DD", value=datetime(2020,10,27))
    end_input = st.date_input(label = "To: ", format="YYYY/MM/DD", value=datetime(2021,10,27))

    model_input = st.selectbox("Model: ", ["Kalman Filter", "Moving Averages"])

    button = st.form_submit_button(label="Go")

with right:
    stat_list = stat_tests()
    correlation = get_pair()[0].corr()[second_stock_input].iloc[0]

    st.markdown(f"""
        <div style='
                    padding: 10px 15px; 
                    border-radius: 5px; 
                    border: 2px solid #cccccc;
                    position: sticky;
                    '>
            <h3>Statistical Measures</h3>
            <ul>
                <h6>Correlation (High > 0.8): {correlation}</h6>
                <h5>P value tests</h5>
                <li>Cointegration: {stat_list[0]}</li>
                <li>Price Spread AD-Fuller Test: {stat_list[3]}</li>
                <li>Price Ratio AD-Fuller Test: {stat_list[4]}</li>
            </ul>
        </div>
    """,
    unsafe_allow_html=True
    )

    with st.expander("How this works", expanded=True):
        st.markdown("""
            Pairs Trading is a quantitative market-neutral strategy that identifies two cointegrated assets.

            Cointegration reflects a long term equilibrium between two non-stationary time series.
            
            This relationship is the blueprint for Mean Reversion Pair Analysis, 
            which assumes that deviations from the historical mean of the price spread/ratio are temporary.

            Therefore, when the spread/ratio widens or narrows beyond it's typical trajectory,
            the expectation is that these prices will revert back to their historical mean.

            These events can generate trading signals, typically shorting the overvalued asset,
            and going long with the undervalued asset.

            This temporary price divergence will revert to the historical mean, closing a potential profit.
        """)

    with st.expander("Kalman Filter Details"):
        st.markdown("""
            In this model,
            
            The Kalman Filter is a recursive algorithm used to estimate dynamic relationships.

            It's used to project the hedge ratio between two assets and track the price spread.

            With a 90 day rolling window, the Kalman Filter continuously updates its estimates with new data.

            Instead of a fixed average, it consistently adjusts it's expectations,
            making it useful for capturing daily subtle deviations in the price spread relationship.

            The algorithm estimates how Stock B moves in relation to Stock A (independent variable),
            creating a predicted price spread of the relationship.

            Then, the deviation of the actual spread vs the predicted spread is used to predict trading signals
            
            Specifics:

            Starts with a neutral assumption, initial state estimates are set to 0, no prior bias

            Low observation covariance, the model trusts that the observed spread is not very noisy

            Low transition covariance, the model assumes small changes in underlying state over time

            90 day rolling window to train estimation
        """)

    with st.expander("Simple Moving Averages"):
        st.markdown("""
            In this model,
            
            Simple Moving Averages is used to analyze deviations in price ratio over a 5 and 14 day window

            The 5 day SMA reflects shorter term movements, calculating the average price ratio over the recent 5 days

            The 14 day SMA provides a more stable view, averaging the price ratio over a 2 week window

            With a backbone of the recent historical trend, a rolling Z score graph can be created using both the
            5 and 14 day SMA alongside the 2 week standard deviation.

            This illustrates how many standard deviations the short term period is from the long term,
            showing when the ratio is unusually high or low, leading to potential trading events
        """)

    with st.expander("Disclaimer !"):
        st.markdown("""
            This model is intended solely for educational and demonstrative purposes.

            It is not financial advice, and users should not make direct investment decisions from this.
            
            While this model showcases the core concepts of cointegration, mean reversion, and the application
            of both the Kalman Filter and Moving Averages,
            
            It does not account for other real-world trading factors such as 
            Risk Management, Slippage, Market Impact, and more.
            
            It is not backtested with lengthy historical trends

            This is simply a demonstration, and a learning experience for my individual understanding.
        """)

with middle:
    st.title("Cointegration Trading Model")
    
    if button:
        pair_analysis()
    else:
        pair_analysis()
