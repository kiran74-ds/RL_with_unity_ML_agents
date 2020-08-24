### Project Details:
For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

<img src="https://github.com/kiran74-ds/RL_with_unity_ML_agents/blob/master/DQN/Banana_ML_agent.gif" width="600" height="400">

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


### Getting Started

To set up your python environment to run the code in this repository, follow the instructions below.

Create (and activate) a new environment with Python 3.6.

Linux or Mac:
conda create --name drl python=3.6
source activate drlnd
Windows:
conda create --name drl python=3.6 
activate drlnd

Clone the repository, and navigate to the python/ folder. Then, install several dependencies.

git clonehttps://github.com/kiran74-ds/RL_with_unity_ML_agents.git
cd RL_with_unity_ML_agents/python
pip install .

Create an IPython kernel for the drlnd environment.
python -m ipykernel install --user --name drl --display-name "drl"
