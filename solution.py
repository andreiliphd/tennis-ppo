# importing libraries
from ma import MA
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

# starting environment
env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

# configuring environment
config = {
    'environment': {
        'state_size': env_info.vector_observations.shape[1],
        'action_size': 2,
        'number_of_agents': 1
    },
    'pytorch': {
        'device': "cpu"
    },
    'hyperparameters': {
        'hidden_size': 64,
        'discount_rate': 0.99,
        'tau': 0.95,
        'gradient_clip': 5,
        'rollout_length': 2048,
        'adam_learning_rate': 3e-4,
        'adam_epsilon': 1e-5,
        'optimization_epochs': 10,
        'ppo_clip': 0.2,
        'entropy_coefficent': 0.01,
        'mini_batch_number': 32,
    }
}

# starting training loop
def ppo(env, brain_name, agents):
    scores = []
    for i in range(4000):
        agents.learn(env, brain_name)
        current_max_reward = agents.act(env, brain_name)
        mean_100 = np.mean(np.array(scores[-100:]))
        scores.append(current_max_reward)
        if mean_100 > 0.5:
            agents.save()
            print('Environment solved in {} episodes. Last mean 100: {}'.format(i + 1, mean_100))
            break
        print('Episode: {} Score of the current episode: {} Last 100 average: {}'.format(i + 1, current_max_reward,
                                                                                         mean_100))
    return scores

# instantiating multi-agent class
agents = MA(config)

# starting training
all_scores = ppo(env, brain_name, agents)

# visualizing rewards
plt.plot(np.arange(1, len(all_scores)+1), all_scores)
plt.ylabel('Rewards')
plt.xlabel('Episode #')
plt.show()