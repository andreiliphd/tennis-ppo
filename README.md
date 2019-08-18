# Tennis - reinforcement learning PPO implementation
============

Tennis is a reinforcement learning algorithm based on [PPO Paper](https://arxiv.org/abs/1707.06347). The environment is provided by Udacity as a part of deep reinfocement learning nanodegree. In this environment, two agents are playing tennis with aim to reach average score of 0.5 over 100 episodes taking maximum score of the episode. 

---

## History

1. The simpliest approach is to make this project using DDPG but I wanted to challenge myself and I have chosen PPO. One of the reasons for choosing PPO is that Amazon AWS Deep Racer has been using PPO as a main reinforcement learning algortihm to run environment. And this choice is not a mistake at all. I enjoyed it very much as it produces very stable training result.

2. I looked at the repos with implementation of tennis environment and I mentioned that there is not much to choose from. I looked at the repos with multi-agent reinforcement learning and the same result. There is not much to choose from. Very complicated project.

3. I rewatched the course and added to my bookmarks [ShangtongZhang](https://github.com/ShangtongZhang/DeepRL) 2017 and Reacher environment by 
[Jeremi Kaczmarczyk](https://github.com/jknthn/reacher-ppo) 2018. I watched these repos, got an idea and reimplemented from scratch as was suggested by Udacity.

4. Happily it is working for two agents.


---



## Features

You can change:
- hidden neurons size
- discount rate
- TAU factor
- gradient clip
- rollout length
- learning rate of an optimizer
- optimizer epsilon
- number of optimization epochs
- clipping PPO value
- entropy coefficient
- mini batch number

---


## Screenshot

Taken from real training. Tennis environment in action. Close to the target:
![Tennis - solving environment](https://github.com/andreiliphd/tennis-ppo/blob/master/images/tennis.gif)

Success!
![Tennis - success](https://github.com/andreiliphd/tennis-ppo/blob/master/images/Screenshot_1.png)


---


## Rewards

![Tennid - rewards](https://github.com/andreiliphd/tennis-ppo/blob/master/images/Figure_1.png)


---

## Setup
1. Clone this repo: 
```
git clone https://github.com/andreiliphd/tennis-ppo.git
```

2. Create and activate a new environment with Python 3.6.
```
conda create --name drlnd python=3.6
conda activate drlnd
```

3. Install PyTorch 0.4.0:
```
conda install pytorch=0.4.0 -c pytorch
```

3. Clone the Udacity repository, and navigate to the python/ folder. Then, install dependencies.

```
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

4. Create an IPython kernel for the `drlnd` environment.
```
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down Kernel menu.

![Change Kernel](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)


6. The rest of dependencies come with `Conda` environment by default.

7. In order to run training in a root of Github repositary run:
```
python solution.py
```
You can see that environment and training started.

---


## Installation

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the directory of GitHub repository files.


## Usage

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.



---

## License
You can check out the full license in the LICENSE file.

This project is licensed under the terms of the **MIT** license.
