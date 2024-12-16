from datetime import datetime
import json
import numpy as np
import pandas as pd
import shinybroker as sb
import time

from utils import (
    get_closest_strike,
    get_peaks_bottoms,
    get_peaks_bottoms_weights,
    get_first_peak_bottom_above_below,
)

def get_ibit_btc_spot_df(option_start_date: datetime, durationStr: str, barSizeSetting: str):
    # Load contracts
    btc_crypto_contract = sb.Contract(json.load(open('contracts/btc_crypto_contract.json')))
    ibit_etf_contract = sb.Contract(json.load(open('contracts/ibit_etf_contract.json')))

    # Fetch expiration/strikes for IBIT
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

    # Fetch historical data for IBIT and BTC
    historical_data_ibit = sb.fetch_historical_data(
        contract=ibit_etf_contract,
        durationStr=durationStr,
        barSizeSetting=barSizeSetting,
    )

    historical_data_bitcoin = sb.fetch_historical_data(
        contract=btc_crypto_contract,
        durationStr=durationStr,
        barSizeSetting=barSizeSetting,
    )

    # Set index to timestamp
    historical_data_ibit['hst_dta'].set_index('timestamp', inplace=True)
    historical_data_bitcoin['hst_dta'].set_index('timestamp', inplace=True)

    # Filter data to start date
    df_ibit = historical_data_ibit['hst_dta']
    df_ibit = df_ibit[df_ibit.index >= option_start_date]
    df_bitcoin = historical_data_bitcoin['hst_dta']
    df_bitcoin = df_bitcoin[df_bitcoin.index >= option_start_date]

    # Merge dataframes
    ibit_btc_spot_df = pd.merge(df_ibit['close'], df_bitcoin['close'], left_index=True, right_index=True, how='inner', suffixes=('_ibit', '_btc'))
    
    return ibit_btc_spot_df, strikes

def append_option_data_to_df(
        ibit_btc_spot_df: pd.DataFrame,
        strikes: list[str],
        lastTradeDateOrContractMonth: str,
        durationStr: str,
        barSizeSetting: str,
    ):
    
    # Load contracts
    ibit_option_contract = json.load(open('contracts/ibit_option_contract.json'))
    
    # Update contracts for calls and puts
    ibit_option_call_contract = ibit_option_contract
    ibit_option_call_contract['right'] = 'C'
    ibit_option_call_contract['lastTradeDateOrContractMonth'] = lastTradeDateOrContractMonth

    ibit_option_put_contract = ibit_option_contract
    ibit_option_put_contract['right'] = 'P'
    ibit_option_put_contract['lastTradeDateOrContractMonth'] = lastTradeDateOrContractMonth

    ibit_call_closes = []
    ibit_put_closes = []

    last_ibit_call_value = 0
    last_ibit_put_value = 0

    stored_dfs = {}

    used_strikes = []

    # Iterate through ETF data
    for _, row in enumerate(ibit_btc_spot_df.iterrows()):

        ibit_close = row[1]['close_ibit']
        strike = get_closest_strike(strikes, ibit_close)
        used_strikes.append(strike)

        if strike in stored_dfs:
            ibit_call_data = stored_dfs[strike]['ibit_call']
            ibit_put_data = stored_dfs[strike]['ibit_put']
        else:
            ibit_option_call_contract['strike'] = strike
            ibit_option_put_contract['strike'] = strike
            
            # Fetch historical data for calls and puts at closest price
            ibit_call_data = sb.fetch_historical_data(
                contract=sb.Contract(ibit_option_call_contract),
                durationStr=durationStr,
                barSizeSetting=barSizeSetting,
            )

            ibit_put_data = sb.fetch_historical_data(
                contract=sb.Contract(ibit_option_put_contract),
                durationStr=durationStr,
                barSizeSetting=barSizeSetting,
            )

            stored_dfs[strike] = {
                'ibit_call': ibit_call_data,
                'ibit_put': ibit_put_data
            }

            time.sleep(1)

        try:
            # Get close value for calls and puts
            ibit_call_value = ibit_call_data['hst_dta'][ibit_call_data['hst_dta']['timestamp'] == row[0]]['close'].values[0]
            ibit_put_value = ibit_put_data['hst_dta'][ibit_put_data['hst_dta']['timestamp'] == row[0]]['close'].values[0]
        except IndexError:
            ibit_call_value = last_ibit_call_value
            ibit_put_value = last_ibit_put_value

        ibit_call_closes.append(ibit_call_value)
        ibit_put_closes.append(ibit_put_value)

        last_ibit_call_value = ibit_call_value
        last_ibit_put_value = ibit_put_value

    ibit_btc_spot_df['ibit_call'] = ibit_call_closes
    ibit_btc_spot_df['ibit_put'] = ibit_put_closes
    ibit_btc_spot_df['strike'] = used_strikes

    return ibit_btc_spot_df

def append_peak_bottom_data_to_df(
        ibit_btc_spot_df: pd.DataFrame,
    ):
    # Load contracts
    btc_crypto_contract = sb.Contract(json.load(open('contracts/btc_crypto_contract.json')))

    # Fetch historical data for BTC
    historical_data_bitcoin_peaks_bottoms = sb.fetch_historical_data(
        contract=btc_crypto_contract,
        durationStr='1 Y',
        barSizeSetting='1 day',
    )

    # Set index to timestamp
    df_bitcoin_peaks_bottoms = historical_data_bitcoin_peaks_bottoms['hst_dta']
    df_bitcoin_peaks_bottoms.set_index('timestamp', inplace=True)

    # Get peaks and bottoms
    peaks, bottoms = get_peaks_bottoms(historical_data_bitcoin_peaks_bottoms['hst_dta']['close'])
    peaks_bottoms = peaks + bottoms

    # Get weights
    weights = get_peaks_bottoms_weights(historical_data_bitcoin_peaks_bottoms['hst_dta']['close'], peaks_bottoms)

    # Create dataframe with peaks, bottoms and weights
    df_weights = pd.DataFrame({
        'peaks_bottoms': peaks_bottoms,
        'weights': weights
    })

    df_weights = df_weights.sort_values(by='peaks_bottoms').reset_index(drop=True)

    ath = ibit_btc_spot_df['close_btc'].max()

    aboves, above_weights, belows, below_weights = [], [], [], []

    # Iterate through BTC data
    for _, row in ibit_btc_spot_df.iterrows():
        close_btc = row['close_btc']
        # Get first peak, bottom, above and below
        above, above_weight, below, below_weight = get_first_peak_bottom_above_below(close_btc, peaks_bottoms, weights)

        if above == None:
            above = ath
            above_weight = below_weight

        aboves.append(above)
        above_weights.append(above_weight)
        belows.append(below)
        below_weights.append(below_weight)

    ibit_btc_spot_df['above'] = aboves
    ibit_btc_spot_df['above_weight'] = above_weights
    ibit_btc_spot_df['below'] = belows
    ibit_btc_spot_df['below_weight'] = below_weights

    return ibit_btc_spot_df

def append_one_year_vol(
        ibit_btc_spot_df: pd.DataFrame,
    ):
    # Load contracts
    btc_crypto_contract = sb.Contract(json.load(open('contracts/btc_crypto_contract.json')))

    # Fetch historical data for BTC
    historical_data_btc = sb.fetch_historical_data(
        contract=btc_crypto_contract,
        durationStr='2 Y',
        barSizeSetting='1 day',
    )

    # Set index to timestamp
    historical_data_btc_df = historical_data_btc['hst_dta']
    historical_data_btc_df['timestamp'] = pd.to_datetime(historical_data_btc_df['timestamp'])

    # Calculate returns
    historical_data_btc_df['close_returns'] = historical_data_btc_df['close'].pct_change()

    vols = []

    # Calculate volatility for the last year
    for t_stop, _ in ibit_btc_spot_df.iterrows():
        t_start = t_stop - pd.DateOffset(years=1)
        vols.append(np.std(historical_data_btc_df[(historical_data_btc_df['timestamp'] < t_stop) & (historical_data_btc_df['timestamp'] > t_start)]['close_returns']))

    ibit_btc_spot_df['volatility'] = vols

    return ibit_btc_spot_df

if __name__ == '__main__':
    option_start_date = datetime(2024, 11, 25)
    durationStr = '3 M'
    barSizeSetting = '1 hour'
    lastTradeDateOrContractMonth = "20241227"

    ibit_btc_spot_df, strikes = get_ibit_btc_spot_df(
        option_start_date=option_start_date,
        durationStr=durationStr,
        barSizeSetting=barSizeSetting
    )

    ibit_btc_spot_df = append_option_data_to_df(
        ibit_btc_spot_df=ibit_btc_spot_df,
        strikes=strikes,
        lastTradeDateOrContractMonth=lastTradeDateOrContractMonth,
        durationStr=durationStr,
        barSizeSetting=barSizeSetting
    )

    ibit_btc_spot_df = append_peak_bottom_data_to_df(
        ibit_btc_spot_df=ibit_btc_spot_df
    )

    ibit_btc_spot_df = append_one_year_vol(
        ibit_btc_spot_df=ibit_btc_spot_df
    )

    ibit_btc_spot_df = ibit_btc_spot_df.round(2)

    ibit_btc_spot_df.to_csv('data/ibit_btc_spot_df.csv')