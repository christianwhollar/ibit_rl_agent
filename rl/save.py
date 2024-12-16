import pickle

def save_agent(agent, model_path='models/model.keras', params_path='params/agent_params.pkl'):
    # Save the trained model
    agent.target_model.save('target_' + model_path)
    agent.model.save(model_path)

    # Save agent parameters
    # Store only the scalar parameters or simple attributes you need to reconstruct or continue training the agent later.
    agent_data = {
        'epsilon': agent.epsilon,
        'epsilon_min': agent.epsilon_min,
        'epsilon_decay': agent.epsilon_decay,
        'gamma': agent.gamma,
        'state_size': agent.state_size,
        'action_size': agent.action_size,
    }

    with open(params_path, 'wb') as f:
        pickle.dump(agent_data, f)
