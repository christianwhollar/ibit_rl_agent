# ibit_rl_agent
*Christian Hollar (christian.hollar@duke.edu)*

## Table of Contents
- [Project Purpose](#project-purpose)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Summary](#summary)

## Project Purpose
This project implements a Reinforcement Learning (RL) agent to trade IBIT options. The primary goal is to detect when market volatility is high or low and act accordingly:
- **High Volatility:** Buy a call/put pair (a "straddle") in anticipation of significant price moves.
- **Low Volatility:** Sell a call/put pair (open a short straddle) to profit from stable price conditions.

The agent always attempts to trade the maximum number of call/put pairs it can afford with the given initial balance. If the agent currently holds a long position (bought calls and puts) and identifies a low volatility scenario, it will sell those positions before opening a short position, and vice versa. This ensures the agent only holds one type of position (long or short) at any given time.

The dataset includes:
- `close_ibit`: IBIT ETF price per timestep
- `close_btc`: Bitcoin spot price (providing market context)
- `ibit_call`, `ibit_put`: Prices of IBIT call and put options
- `strike`: The strike price for the options
- `above`, `below`, `above_weight`, `below_weight`: Horizontal Support and Resistance (HSAR) lines and their frequency weights, indicating how often the price has interacted with these levels
- `volatility`: A 1-year trailing volatility measure

Note that IBIT options have only been traded for about one month, resulting in a very limited historical dataset. Thus, any performance results should be viewed with caution due to the small sample size.

### The DQN Agent and Trading Environment
The RL agent is built using a Deep Q-Network (DQN) approach:
- The agent receives an observation of the current market state, including prices, option values, HSAR levels, volatility, and current positions.
- At each timestep, it selects an action: Sit (do nothing), Buy (open or close short position), or Sell (open or close long position).
- The environment (a `TradingEnv` class based on Gym) updates the agent’s portfolio, computes rewards based on changes in portfolio value, and moves to the next timestep.

Over time, the DQN learns a policy that aims to maximize long-term returns by choosing actions aligned with predicted market volatility conditions.

## How to Run
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Run `app.py` if you want the Shinybroker interface, or run training and testing scripts directly in Python. For example:
```python
import pandas as pd
from rl.test import test_agent
from rl.train import train_dqn_agent
from rl.predict import predict_action
from rl.save import save_agent
from rl.load import load_agent

# Load your data
df = pd.read_csv('data/ibit_btc_spot_df.csv')
train_size = int(len(df)*0.9)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# Train the agent
trained_agent = train_dqn_agent(train_data, episodes=10, window_size=5, batch_size=64)
save_agent(trained_agent)

# Test the agent
final_value, profit, summary_df = test_agent(trained_agent, test_data, window_size=1)
print(summary_df)
print(f"Final Portfolio Value on Test Data: {final_value}")
print(f"Profit on Test Data: {profit}")

# Predict action for a single row
trained_agent = load_agent()
action_str = predict_action(trained_agent, df, row_index=0)
print(f"Recommended action for row 0: {action_str}")
```

## Repository Structure
```
/IBIT_RL_AGENT/
├── app.py                   <- Shinybroker interface (can be run in Python)
├── build_df.py              <- Constructs the input DataFrame with all required columns
├── utils.py                 <- Utility functions
├── demo.ipynb               <- Jupyter notebook demonstration
├── contracts/               <- JSON contract definitions
├── data/
│   └── ibit_btc_spot_df.csv <- Historical input data
├── models/                  <- Stored RL models
├── target_models/           <- Stored RL target models
├── params/                  <- RL parameters
├── rl/
│   ├── __init__.py
│   ├── agent.py             <- DQNAgent class
│   ├── env.py               <- TradingEnv class
│   ├── train.py             <- Training functions
│   ├── test.py              <- Testing functions
│   ├── load.py              <- Load trained agent
│   ├── save.py              <- Save trained agent
│   ├── predict.py           <- Predict actions from trained agent
│   └── ibit_option_contract.py
```

## Summary
The Reinforcement Learning agent uses a DQN to forecast when to buy or sell call/put pairs. By estimating future volatility from the provided features, the agent tries to capitalize on high volatility by opening a long position (buying calls and puts) or benefit from stable periods by opening a short position (selling calls and puts).

While limited historical data restricts the agent’s capacity to generalize, preliminary tests showed the agent performing best with 8 episodes of training and a window size of 5. During testing, it achieved about a 4% return in 7 days, depending on the initial balance. This result, however, should not be over-interpreted given the small data sample (only one month of IBIT option trading history).
