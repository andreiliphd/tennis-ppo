# importing libraries
from model import PPO
from agent import SinglePPOAgent
import numpy as np
import torch

# two agent class: main control point
class MA:
    # instanstiating needed attributes of the class
    def __init__(self, config):
        self.config = config
        self.ap1 = PPO(config)
        self.ap2 = PPO(config)
        self.agent1 = SinglePPOAgent(self.ap1, config)
        self.agent2 = SinglePPOAgent(self.ap2, config)

    # play no training
    def act(self, env, brain_name):
        env_info = env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        states1 = states[0]
        states2 = states[1]
        scores = np.zeros(2)
        while True:
            actions1, log_probs1, _, values1 = self.ap1(states1)
            actions2, log_probs2, _, values2 = self.ap2(states2)
            actions = torch.cat((actions1, actions2), dim=0)
            env_info = env.step([actions.cpu().detach().numpy()])[brain_name]
            next_states = env_info.vector_observations
            dones = env_info.local_done
            scores += env_info.rewards
            states = next_states
            states1 = states[0]
            states2 = states[1]
            if np.any(dones):
                break

        return np.max(scores)

    # training
    def learn(self, env, brain_name):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        states1 = states[0]
        states2 = states[1]

        rollout1 = []
        rollout2 = []

        for k in range(self.config['hyperparameters']['rollout_length']):
            terminals = np.array([])
            actions1, log_probs1, _ , values1 = self.ap1(states1)
            actions2, log_probs2, _, values2 = self.ap2(states2)
            actions = torch.cat((actions1, actions2), dim=0)
            env_info = env.step([actions.cpu().detach().numpy()])[brain_name]
            next_states1 = env_info.vector_observations[0]
            next_states2 = env_info.vector_observations[1]
            rewards = env_info.rewards
            rewards1 = np.array(rewards[0]).reshape([1])
            rewards2 = np.array(rewards[1]).reshape([1])
            if np.any(env_info.local_done):
                terminals = np.append(terminals, np.array(1))
            else:
                terminals = np.append(terminals, np.array(0))

            rollout1.append(
                [states1, values1.detach(), actions1.detach(), log_probs1.detach(), rewards1, 1 - terminals])
            rollout2.append(
                [states2, values2.detach(), actions2.detach(), log_probs2.detach(), rewards2, 1 - terminals])
            states1 = next_states1
            states2 = next_states2
        pending_value1 = self.ap1(states1)[-1]
        pending_value2 = self.ap2(states2)[-1]
        rollout1.append([states1, pending_value1, None, None, None, None])
        rollout2.append([states2, pending_value2, None, None, None, None])
        self.agent1.step(rollout1, pending_value1)
        self.agent2.step(rollout2, pending_value2)

    def save(self):
        torch.save(self.ap1.state_dict(),
                   f"ap1.pth")
        torch.save(self.ap2.state_dict(),
                   f"ap2.pth")

