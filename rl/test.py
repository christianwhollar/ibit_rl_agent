from rl.env import TradingEnv
import pandas as pd

def test_agent(agent, test_df, window_size=1):
    env = TradingEnv(data=test_df, initial_balance=10000, window_size=window_size)
    observation, _ = env.reset()
    done = False

    steps_data = {
        'step': [],
        'action': [],
        'balance': [],
        'open_order_value': [],
        'calls_open': [],
        'puts_open': [],
        'strike_price': []
    }

    step_count = 0
    while not done:
        action = agent.act(observation, exploit_only=True)
        # record state before step
        steps_data['step'].append(step_count)
        steps_data['action'].append(action)
        steps_data['balance'].append(env.balance)
        steps_data['open_order_value'].append(env.get_open_order_value())
        calls_open, puts_open = env.get_calls_and_puts_open()
        steps_data['calls_open'].append(calls_open)
        steps_data['puts_open'].append(puts_open)
        steps_data['strike_price'].append(env.entry_strike)

        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_count += 1

    final_value = env._get_portfolio_value()
    profit = final_value - env.initial_balance

    summary_df = pd.DataFrame(steps_data)
    return final_value, profit, summary_df