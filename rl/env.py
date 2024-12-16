import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        data: pd.DataFrame, 
        initial_balance=10000, 
        window_size=1
    ):
        super().__init__()

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.window_size = window_size

        self.current_step = 0
        self.max_steps = len(self.data) - 1

        self.balance = initial_balance
        self.position_type = 0  # 0=no position, 1=long, -1=short
        self.n_pairs = 0
        self.entry_ibit_price = 0.0
        self.entry_strike = 0.0
        self.prev_portfolio_value = self.initial_balance

        # Actions: 0 = SIT, 1 = BUY (or close short), 2 = SELL (or close long)
        self.action_space = spaces.Discrete(3)

        # Observation: 
        # columns used: close_ibit, close_btc, ibit_call, ibit_put, strike, above, above_weight, below, below_weight, volatility
        # plus balance, position_type, n_pairs
        # total obs dim = 10 from df + 3 = 13
        obs_shape = (13,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position_type = 0
        self.n_pairs = 0
        self.entry_ibit_price = 0.0
        self.entry_strike = 0.0
        self.prev_portfolio_value = self._get_portfolio_value()

        return self._get_observation(), {}

    def step(self, action: int):
        # Current market data
        row = self.data.iloc[self.current_step]
        close_ibit = row['close_ibit']
        ibit_call = row['ibit_call']
        ibit_put = row['ibit_put']
        strike = row['strike']

        done = False
        info = {}

        # Determine feasibility of actions
        cost_per_pair = (ibit_call + ibit_put) + close_ibit

        # Execute action
        if self.position_type == 0:
            # no position
            if action == 1 and cost_per_pair > 0:
                # open long
                max_pairs = self.balance // cost_per_pair
                if max_pairs > 0:
                    self.n_pairs = int(max_pairs)
                    total_cost = self.n_pairs * cost_per_pair
                    self.balance -= total_cost
                    self.position_type = 1
                    self.entry_ibit_price = close_ibit
                    self.entry_strike = strike

            elif action == 2 and cost_per_pair > 0:
                # open short
                max_pairs = self.balance // cost_per_pair
                if max_pairs > 0:
                    self.n_pairs = int(max_pairs)
                    total_credit = self.n_pairs * cost_per_pair
                    self.balance += total_credit
                    self.position_type = -1
                    self.entry_ibit_price = close_ibit
                    self.entry_strike = strike

        elif self.position_type == 1:
            # long position
            if action == 2:
                # close long
                cost_to_close = (ibit_call + ibit_put + close_ibit) * self.n_pairs
                self.balance += cost_to_close
                self.n_pairs = 0
                self.position_type = 0
                self.entry_ibit_price = 0.0
                self.entry_strike = 0.0

        elif self.position_type == -1:
            # short position
            if action == 1:
                # close short
                cost_to_close = (ibit_call + ibit_put + close_ibit) * self.n_pairs
                self.balance -= cost_to_close
                self.n_pairs = 0
                self.position_type = 0
                self.entry_ibit_price = 0.0
                self.entry_strike = 0.0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        current_portfolio_value = self._get_portfolio_value()
        reward = current_portfolio_value - self.prev_portfolio_value
        self.prev_portfolio_value = current_portfolio_value

        return self._get_observation(), reward, done, False, info

    def _get_portfolio_value(self):
        if self.position_type == 0:
            return self.balance
        else:
            # current data
            row = self.data.iloc[self.current_step]
            close_ibit = row['close_ibit']
            ibit_call = row['ibit_call']
            ibit_put = row['ibit_put']
            cost_per_pair = (ibit_call + ibit_put + close_ibit)
            if self.position_type == 1:
                return self.balance + cost_per_pair * self.n_pairs
            else:
                # short
                return self.balance - cost_per_pair * self.n_pairs

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        obs = [
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
            self.balance,
            self.position_type,
            self.n_pairs
        ]

        return np.array(obs, dtype=np.float32)

    def get_open_order_value(self):
        if self.position_type == 0:
            return 0.0
        # current data
        row = self.data.iloc[self.current_step]
        close_ibit = row['close_ibit']
        ibit_call = row['ibit_call']
        ibit_put = row['ibit_put']
        cost_per_pair = (ibit_call + ibit_put + close_ibit)
        # Return absolute current value of the open order
        return abs(cost_per_pair * self.n_pairs)

    def get_calls_and_puts_open(self):
        # For a long position: calls_open = n_pairs, puts_open = n_pairs
        # For a short position: calls_open = -n_pairs, puts_open = -n_pairs
        # For no position: 0
        if self.position_type == 1:
            return self.n_pairs, self.n_pairs
        elif self.position_type == -1:
            return -self.n_pairs, -self.n_pairs
        else:
            return 0, 0
