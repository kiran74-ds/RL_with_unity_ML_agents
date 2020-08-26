### Project Details:
For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

<img src="https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/DQN/images/Banana_ML_agent.gif" width="600" height="400">

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

+ 0 - move forward.
+ 1 - move backward.
+ 2 - turn left.
+ 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


### Dependenices

Please follow the instructions mentioned in the main page 


### Instructions

To run the code, after sucessfully created virtual environment and kernel, open the jupyter notebook located at https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/DQN/code/Navigation.ipynb and follow the notebook.

```
# instantiate the Agent class
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

# Run the function dqn(),Agent starts learning here and returns the rewards for each episode
scores = dqn(1000) #1000 here is number of episodes
```
