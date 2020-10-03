### Project Details:
For this project, you will work with the Tennis environment.

<img src="https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/Tennis_MADDPG/images/Tennis.gif" width="600" height="400">

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5


### Dependenices

Please follow the instructions mentioned in the main page 


### Instructions

Follow the below steps to run the code:

+ Create Virtual Environment, Install Packages and Create IPython Kernel as mentioned in the main Readme.md file
+ Open the jupyter notebook located at https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/Tennis_MADDPG/code/Tennis.ipynb
+ Select the kernel that you just created
+ Follow the steps in the note book 

```

# Run the function ddpg(),Agents starts learning here and returns the rewards which is maximum of these two agents rewards for each episode 
scores = ddpg(30000) #3000 is the number of episodes
```
