import numpy as np
from model import Actor, Critic
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

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
#Enable cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Main DDPG agent that extracts experiences and learns from them"""
    def __init__(self, state_size=24, action_size=2, random_seed=0):
        """
        Initializes Agent object.
        @Param:
        1. state_size: dimension of each state.
        2. action_size: number of actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        #Actor network
        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #Critic network
        self.critic_local = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        #Noise proccess
        self.noise = OUNoise(action_size, random_seed) #define Ornstein-Uhlenbeck process

        #Replay memory
        self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, MINI_BATCH, random_seed) #define experience replay buffer object
        
        self.time_step = 0 
        
    def step(self, state, action, reward, next_state, done):
        """
        Saves an experience in the replay memory to learn from using random sampling.
        @Param:
        1. state: current state, S.
        2. action: action taken based on current state.
        3. reward: immediate reward from state, action.
        4. next_state: next state, S', from action, a.
        5. done: (bool) has the episode terminated?
        Exracted version for trajectory used in calculating the value for an action, a."""
        self.time_step += 1
        self.memory.add(state, action, reward, next_state, done) #append to memory buffer

        # only learn every n_time_steps
        if self.time_step % UPDATE_TIME_STEPS != 0:
            return

        #check if enough samples in buffer. if so, learn from experiences, otherwise, keep collecting samples.
        if(len(self.memory) > MINI_BATCH):
            for _ in range(N_LEARN_UPDATES):
                experience = self.memory.sample()
                self.learn(experience)

    def reset(self):
        """Resets the noise process to mean"""
        self.noise.reset()

    def act(self, state, add_noise=True):
        """
        Returns a deterministic action given current state.
        @Param:
        1. state: current state, S.
        2. add_noise: (bool) add bias to agent, default = True (training mode)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) #typecast to torch.Tensor
        self.actor_local.eval() #set in evaluation mode
        with torch.no_grad(): #reset gradients
            action = self.actor_local(state).cpu().data.numpy() #deterministic action based on Actor's forward pass.
        self.actor_local.train() #set training mode

        #If training mode, i.e. add_noise = True, add noise to the model to learn a more accurate policy for current state.
        if(add_noise):
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def learn(self, experiences, gamma=GAMMA):
        """
        Learn from a set of experiences picked up from a random sampling of even frequency (not prioritized)
        of experiences when buffer_size = MINI_BATCH.
        Updates policy and value parameters accordingly
        @Param:
        1. experiences: (Tuple[torch.Tensor]) set of experiences, trajectory, tau. tuple of (s, a, r, s', done)
        2. gamma: immediate reward hyper-parameter, 0.99 by default.
        """
        #Extrapolate experience into (state, action, reward, next_state, done) tuples
        states, actions, rewards, next_states, dones = experiences

        #Update Critic network
        actions_next = self.actor_target(next_states) # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) #  r + γ * Q-values(a,s)

        # Compute critic loss using MSE
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) #clip gradients
        self.critic_optimizer.step()

        #Update Actor Network

        # Compute actor loss
        actions_pred = self.actor_local(states) #gets mu(s)
        actor_loss = -self.critic_local(states, actions_pred).mean() #gets V(s,a)
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters. Copies model τ every experience.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer():
    """
    Implementation of a fixed size replay buffer as used in DQN algorithms.
    The goal of a replay buffer is to unserialize relationships between sequential experiences, gaining a better temporal understanding.
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initializes the buffer.
        @Param:
        1. action_size: env.action_space.shape[0]
        2. buffer_size: Maximum length of the buffer for extrapolating all experiences into trajectories. default - 1e6 (Source: DeepMind)
        3. batch_size: size of mini-batch to train on. default = 64.
        """
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.replay_memory = deque(maxlen=buffer_size) #Experience replay memory object
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) #standard S,A,R,S',done
        
    def add(self, state, action, reward, next_state, done):
        """Adds an experience to existing memory"""
        trajectory = self.experience(state, action, reward, next_state, done)
        self.replay_memory.append(trajectory)
    
    def sample(self):
        """Randomly picks minibatches within the replay_buffer of size mini_batch"""
        experiences = random.sample(self.replay_memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):#override default __len__ operator
        """Return the current size of internal memory."""
        return len(self.replay_memory)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state 