from rl.agent import DQNAgent
from tensorflow.keras.models import load_model
import pickle

def load_agent(model_path='models/model.keras', params_path='params/agent_params.pkl'):
    
    # Load the parameters
    with open(params_path, 'rb') as f:
        agent_data = pickle.load(f)

    # Create a new agent with the same state_size and action_size
    new_agent = DQNAgent(
        state_size=agent_data['state_size'],
        action_size=agent_data['action_size']
    )

    # Load the model weights
    new_agent.model = load_model(model_path, compile=False)
    new_agent.target_model = load_model('target_' + model_path, compile=False)

    # Restore the parameters
    new_agent.epsilon = agent_data['epsilon']
    new_agent.epsilon_min = agent_data['epsilon_min']
    new_agent.epsilon_decay = agent_data['epsilon_decay']
    new_agent.gamma = agent_data['gamma']

    return new_agent
