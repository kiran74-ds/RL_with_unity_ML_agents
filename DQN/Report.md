
### Deep Q-Networks Algorithms

Network Architecture
```
QNetwork(
  (fc1): Linear(in_features=37, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=4, bias=True)
)
```
Hyperperparameters Used:

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

### Plot of Rewards

Agent solved the enviroment in 700 episodes.


### Ideas for Future Work

This model can be further improved by using Double DQN, Prioritized Experience Replay, Duelling DQN and Rainbow algorithms.



