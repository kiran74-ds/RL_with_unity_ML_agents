
### Deep Determinstic Policy Gradient 


+ To Solve any Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments in this scenario Tennis, we can use Multi Agent Deep Determinstic Policy Gradient(MADDPG).
+ Multi-agent DDPG, extends DDPG into a multi-agent policy gradient algorithm where decentralized agents learn a centralized critic based on the observations and actions of all agents.
+ The critic is augmented with extra information about the policies of other agents, while the actor only has access to local information. After training is completed, only the local actors are used at execution phase, acting in a decentralized manner.

Network Architecture

Actor
```
Actor(
  (fc1): Linear(in_features=24, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
```

Critic

```
Critic(
  (fc1): Linear(in_features=24, out_features=256, bias=True)
  (fc2): Linear(in_features=258, out_features=128, bias=True)
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
```

### Plot of Rewards

Agents solved the enviroment in 2207 episodes 

<img src="https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/Tennis_MADDPG/images/tennis_performance.png" width=600, height=400>


### Ideas for Future Work

This model can be further improved by 
+ Tuning hyperparameters 
+ Running network for longer time 
+ Changing network Architecture to have different architecute for each agent 
+ Implementing other algorithms such as PPO, A3C 


