from rl.agent import DQNAgent
from rl.env import TradingEnv

def train_dqn_agent(df, episodes=50, window_size=1, batch_size=64):
    env = TradingEnv(data=df, initial_balance=10000, window_size=window_size)
    observation, _ = env.reset()
    state_size = len(observation)
    action_size = env.action_space.n

    agent = DQNAgent(state_size=state_size, action_size=action_size)

    for episode in range(episodes):
        observation, _ = env.reset()
        state = observation
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_observation, done)
            state = next_observation
            total_reward += reward

            agent.replay(batch_size)

        agent.update_target_model()
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    return agent