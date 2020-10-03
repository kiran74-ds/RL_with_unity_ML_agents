### Project Details:
For this project, you will train a double-jointed arm can move to target locations

<img src="https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/Continous_Control/images/single_agent.gif" width="600" height="400">

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
 + Vector Observation space size (per agent): 33
 + Vector Action space size (per agent): 4


I am implementing this project for wo separate versions of the Unity environment:
+ The first version contains a single agent.
+ The second version contains 20 identical agents, each with its own copy of the environment.


### Dependenices

Please follow the instructions mentioned in the main page 


### Instructions

Follow the below steps to run the code:

+ Create Virtual Environment, Install Packages and Create IPython Kernel as mentioned in the main Readme.md file
+ Open the jupyter notebook located at
  - For Single Agent: https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/Continous_Control/code/Continous_control_DDPG_single_agent.ipynb
  - For Multi Agents: https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/Continous_Control/code/Continous_control_DDPG_multi_agent.ipynb
+ Select the kernel that you just created
+ Follow the steps in the note book 

```

# Run the function ddpg(),Agent starts learning here and returns the rewards for each episode
scores = ddpg(300) #300 is the number of episodes
```
