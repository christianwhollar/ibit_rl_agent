import shinybroker as sb
import plotly.graph_objects as go
from datetime import datetime, time
import pytz
from shiny import Inputs, Outputs, Session, reactive, ui, render
from shiny.types import SilentException
from shinywidgets import output_widget, render_plotly
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from rl.predict import predict_action
from rl.load import load_agent

step_3_ui = ui.page_fluid(
    ui.row(
        ui.h5("IBIT ETF and Options Analysis"),
    ),
    ui.row(
        ui.output_plot("render_btc_plot"),
    ),
    ui.row(
        ui.output_plot("render_ibit_etf_plot"),
    ),
    ui.row(
        ui.output_plot("render_ibit_call_plot"),
    ),
    ui.row(
        ui.output_plot("render_ibit_put_plot"),
    ),
    ui.row(
        ui.h5("Model Input Data"),
    ),
    ui.row(
        ui.output_data_frame("render_historical_dataframe"),
    ),
    ui.row(
        ui.h5("Model Prediction"),
    ),
    ui.row(
        ui.output_text("render_model_prediction"),
    ),
)


def server(input: Inputs, output: Outputs, session: Session, ib_socket, sb_rvs):
    # Get the BTC Dataframe
    @reactive.calc
    def get_btc_dataframe():
        historical_data = sb.fetch_historical_data(
            sb.Contract({
            "symbol": "BTC",
            "secType": "CRYPTO",
            "currency": "USD",
            "exchange": "PAXOS"
        }),
            durationStr="3 M",
            barSizeSetting="1 hour",
        )

        return historical_data['hst_dta']

    # Get the IBIT ETF Dataframe
    @reactive.calc
    def get_ibit_etf_dataframe():
        historical_data = sb.fetch_historical_data(
            sb.Contract({
                "symbol": "IBIT",
                "secType": "STK",
                "exchange": "SMART",
                "currency": "USD"
            }),
            durationStr="3 M",
            barSizeSetting="1 hour",
        )

        return historical_data['hst_dta']

    # Get the Closest Strike for IBIT
    @reactive.calc
    def get_closest_strike() -> float:
        # strikes = get_strikes()
        # strikes = [float(strike) for strike in strikes]
        # value = float(get_ibit_etf_dataframe()['close'].values[0])
        # return min(strikes, key=lambda x:abs(x-value))
        return 60.0

    # Get the Strikes for IBIT
    @reactive.calc
    def get_strikes():
        matching_signals_ibit = sb.fetch_matching_symbols("IBIT")
        underlyingConId = matching_signals_ibit['stocks'].iloc[0]['con_id']
        underlyingSymbol = matching_signals_ibit['stocks'].iloc[0]['symbol']
        underlyingSecType = matching_signals_ibit['stocks'].iloc[0]['sec_type']

        sec_def_opt_params = sb.fetch_sec_def_opt_params(
            underlyingConId=underlyingConId,
            underlyingSymbol=underlyingSymbol,
            underlyingSecType=underlyingSecType
        )

        strikes = sec_def_opt_params.iloc[0]['strikes'].split(',')

        return strikes

    # Get the Call Bid/AsK for IBIT
    @reactive.calc
    def get_ibit_call_dataframe():
        df_ibit_etf = get_ibit_etf_dataframe()
        strikes = get_strikes()
        value = df_ibit_etf['close'].iloc[-1]

        closest_strike = get_closest_strike()

        historical_data = sb.fetch_historical_data(
            sb.Contract({
                "symbol": "IBIT",
                "secType": "OPT",
                "exchange": "SMART",
                "currency": "USD",
                "lastTradeDateOrContractMonth": "20241227",
                "strike": f"{closest_strike}",
                "right": "C",
                "multiplier": "100"
            }),
            durationStr="3 M",
            barSizeSetting="1 hour",
        )

        return historical_data['hst_dta']
    
    # Get the Put Bid/Ask for IBIT
    @reactive.calc
    def get_ibit_put_dataframe():
        df_ibit_etf = get_ibit_etf_dataframe()
        strikes = get_strikes()
        value = df_ibit_etf['close'].iloc[-1]

        closest_strike = get_closest_strike()

        historical_data = sb.fetch_historical_data(
            sb.Contract({
                "symbol": "IBIT",
                "secType": "OPT",
                "exchange": "SMART",
                "currency": "USD",
                "lastTradeDateOrContractMonth": "20241227",
                "strike": f"{closest_strike}",
                "right": "P",
                "multiplier": "100"
            }),
            durationStr="3 M",
            barSizeSetting="1 hour",
        )

        return historical_data['hst_dta']

    # Get the Input Data for RL Agent
    @reactive.calc
    def get_historical_dataframe():
        df = pd.read_csv('data/ibit_btc_spot_df.csv')
        df = df.iloc[::-1].reset_index(drop=True)
        return df.head(5)

    # Render IBIT ETF Plot
    @render.plot
    def render_ibit_etf_plot():
        df = get_ibit_etf_dataframe()
        fig, ax = plt.subplots()
        ax.plot(df['timestamp'], df['close'])
        ax.set_title('IBIT ETF')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        return fig
    
    # Render IBIT Call Plot
    @render.plot
    def render_ibit_call_plot():
        df = get_ibit_call_dataframe()
        fig, ax = plt.subplots()
        ax.plot(df['timestamp'], df['close'])
        ax.set_title('IBIT Call Option')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        return fig
    
    # Render IBIT Put Plot
    @render.plot
    def render_ibit_put_plot():
        df = get_ibit_put_dataframe()
        fig, ax = plt.subplots()
        ax.plot(df['timestamp'], df['close'])
        ax.set_title('IBIT Put Option')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        return fig
    
    # Render Input Data for RL Agent
    @render.data_frame
    def render_historical_dataframe():
        return get_historical_dataframe()
    
    # Get Model Prediction
    @reactive.calc
    def get_model_prediction():
        df = get_historical_dataframe()
        time = df['timestamp'].iloc[0]
        trained_agent = load_agent()
        action_str = predict_action(trained_agent, df, row_index=0)
        return f"Recommended action for {time}: {action_str} Call and Put Option at Strike Price {get_closest_strike()}"
    
    # Render Model Prediction
    @render.text
    def render_model_prediction():
        return get_model_prediction()

    # Render BTC Plot
    @render.plot
    def render_btc_plot():
        df = get_btc_dataframe()
        fig, ax = plt.subplots()
        ax.plot(df['timestamp'], df['close'])
        ax.set_title('BTC Spot')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        return fig

app = sb.sb_app(
    home_ui=step_3_ui,
    server_fn=server,
    host='127.0.0.1',
    port=7497,
    client_id=10799,
    verbose=True
)

app.run()
