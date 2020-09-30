
### Deep Q-Networks Algorithms

Network Architecture

Actor
```
Actor(
  (fc1): Linear(in_features=33, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
)
```

Critic

```
Critic(
  (fc1): Linear(in_features=33, out_features=256, bias=True)
  (fc2): Linear(in_features=260, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```

Hyperperparameters Used:

```
## Hyper parameters
LR_CRITIC = 1e-4 
LR_ACTOR = 1e-4 
GAMMA = 0.99 
WEIGHT_DECAY = 0 
TAU = 1e-3 
BUFFER_SIZE = int(1e6) 
MINI_BATCH = 128 

N_LEARN_UPDATES = 10         
UPDATE_TIME_STEPS = 20 
```

### Plot of Rewards

Agent solved the enviroment in 60 episodes for single agent

<img src="https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/Continous_control/images/continous_control_single_agent.png" width=600, height=400>


Agent solved the enviroment in 200 episodes for multi agent


<img src="https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/DQN/images/continous_control_multi_agent.png" width=600, height=400>

### Ideas for Future Work

This model can be further improved by 
	+ Tuning hyperparameters 
	+ Running network for longer time 
	+ Changing network Architecture
	+ Implementing other algorithms such as PPO, A3C


