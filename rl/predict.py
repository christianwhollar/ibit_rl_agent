import pandas as pdd
import numpy as np
def predict_action(agent, df, row_index, balance=10000, position_type=0, n_pairs=0):
    """
    Given a trained agent, a dataframe with market data, and a single row index,
    along with current state (balance, position_type, n_pairs),
    return the recommended action as a string: 'buy', 'sell', or 'sit'.
    """
    row = df.iloc[row_index]
    # Construct observation as env does:
    # obs = [close_ibit, close_btc, ibit_call, ibit_put, strike, above, above_weight, below, below_weight, volatility, 
    #        balance, position_type, n_pairs]
    obs = np.array([
        row['close_ibit'],
        row['close_btc'],
        row['ibit_call'],
        row['ibit_put'],
        row['strike'],
        row['above'],
        row['above_weight'],
        row['below'],
        row['below_weight'],
        row['volatility'],
        balance,
        position_type,
        n_pairs
    ], dtype=np.float32)

    action = agent.act(obs, exploit_only=True)
    if action == 0:
        return "sit"
    elif action == 1:
        return "buy"
    else:
        return "sell"