#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Jeremi Kaczmarczyk (jeremi.kaczmarczyk@gmail.com) 2018
# For Udacity Deep Reinforcement Learning Nanodegree

# Modified by Andrei Li (andreiliphd@gmail.com) 2019
# For Udacity Deep Reinforcement Learning Nanodegree

# importing libraries
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Single aggent
class SinglePPOAgent(object):
    
    # class initialization
    def __init__(self, ap, config):
        self.config = config
        self.ap = ap
        self.optimizer = Adam(ap.parameters(), config['hyperparameters']['adam_learning_rate'],
                              eps=config['hyperparameters']['adam_epsilon'])

    # step of the training
    # details of an implementation in Report.ipynb
    def step(self, rollout, pending_value):
        rout_ready = [None] * (len(rollout) - 1)


        advantages = torch.Tensor(np.zeros((1, 1)))
        returns = pending_value.detach()
        for i in reversed(range(len(rollout) - 1)):
            states, value, actions, log_probs, rewards, dones = rollout[i]
            dones = torch.Tensor(dones).unsqueeze(1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            actions = torch.Tensor(actions)
            states = torch.Tensor(states)
            next_value = rollout[i + 1][1]
            returns = rewards + self.config['hyperparameters']['discount_rate'] * dones * returns

            td_error = rewards + self.config['hyperparameters']['discount_rate'] * dones * next_value.detach() - value.detach()
            advantages = advantages * self.config['hyperparameters']['tau'] * self.config['hyperparameters']['discount_rate'] * dones + td_error
            rout_ready[i] = [states.reshape([1,24]), actions, log_probs, returns, advantages]


        states, actions, log_probs_after_processing, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*rout_ready))
        advantages = (advantages - advantages.mean()) / advantages.std()
        indecies = []
        rl = list(range(0, self.config['hyperparameters']['rollout_length']))
        random.shuffle(rl)
        for num in range(0, len(rl)//64):
            indecies.append(rl[num * 64:num*64 + 64])

        for batch_indices in indecies:
            sampled_states = states[batch_indices]
            sampled_actions = actions[batch_indices]
            sampled_log_probs_old = log_probs_after_processing[batch_indices]
            sampled_returns = returns[batch_indices]
            sampled_advantages = advantages[batch_indices]

            _, log_probs, entropy_loss, values = self.ap(sampled_states, sampled_actions)
            ratio = (log_probs - sampled_log_probs_old).exp()
            obj = ratio * sampled_advantages
            obj_clipped = ratio.clamp(1.0 - self.config['hyperparameters']['ppo_clip'],
                                      1.0 + self.config['hyperparameters']['ppo_clip']) * sampled_advantages
            policy_loss = -torch.min(obj, obj_clipped).mean(0) - self.config['hyperparameters']['entropy_coefficent'] * entropy_loss.mean()

            value_loss = 0.5 * (sampled_returns - values).pow(2).mean()

            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(self.ap.parameters(), self.config['hyperparameters']['gradient_clip'])
            self.optimizer.step()

