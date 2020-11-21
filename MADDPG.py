import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from noise import OUNoise
from replay_buffer import ReplayBuffer


import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size   ###
BATCH_SIZE = 128        # minibatch size      
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor         ###
LR_CRITIC = 1e-3        # learning rate of the critic        ###
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # update the actor and critic networks every 'UPDATE_EVERY' time steps   ### 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Replay memory
memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        
        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        
        # Count number of steps
        self.n_steps = 0                               ###
    
    def step(self, num_agent):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # memory.add(state, action, reward, next_state, done)

        self.n_steps = (self.n_steps + 1) % UPDATE_EVERY                  ###
        # Learn, if enough samples are available in memory
        if len(memory) > BATCH_SIZE and self.n_steps == 0:         ###
            experiences = memory.sample()
            self.learn(experiences, GAMMA, num_agent)
     
        self.n_steps += 1

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
             action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        
        return np.clip(action, -1, 1)

    def reset(self):
        self.n_steps = 0
        self.noise.reset()

    def learn(self, experiences, gamma, num_agent):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        state, actions, rewards, next_state, dones = experiences
        
         # Splits states into a 'num_agents' of states
        all_states = torch.chunk(state, 2, dim = 1)
        
         # Splits 'next_x' into a 'num_agents' of next states
        all_next_states = torch.chunk(next_state, 2, dim = 1)
        
         # Get reward for each agent
        rewards = rewards[:,num_agent].reshape(rewards.shape[0],1)
        dones = dones[:,num_agent].reshape(dones.shape[0],1)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = [self.actor_target(n_s) for n_s in all_next_states]
        target_actions = torch.cat(actions_next, dim=1).to(device) 
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_state, target_actions)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(state, actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
      
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [self.actor_local(s) for s in all_states]
        actions_pred_ = torch.cat(actions_pred, dim=1).to(device)
        
        actor_loss = -self.critic_local(state, actions_pred_).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
class MultiAgent:
    """Interaction between multiple agents in common environment"""
    def __init__(self, state_size, action_size, num_agents, seed = 0):
        self.state_size = state_size
        self.action_size = action_size        
        self.num_agents = num_agents
        self.seed = seed
        self.agents = [Agent(self.state_size, self.action_size, self.seed) for x in range(self.num_agents)]
        
    def step(self, state, actions, rewards, next_state, dones):
        """Save experiences in replay memory and learn."""
        # Save experience in replay memory
        memory.add(state, actions, rewards, next_state, dones)

        for num_agent, agent in enumerate(self.agents):
            agent.step(num_agent)

    def act(self, states, add_noise=True):
        """Agents perform actions according to their policy."""
        actions = np.zeros([self.num_agents, self.action_size])
        for index, agent in enumerate(self.agents):
            actions[index, :] = agent.act(states[index], add_noise)
        return actions
    
    
    def reset(self):        
        for agent in self.agents:
            agent.reset()
           
    def save_model(self):
        """Save learnable model's parameters of each agent."""
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'agent{}_checkpoint_actor.pth'.format(index + 1))
            torch.save(agent.critic_local.state_dict(), 'agent{}_checkpoint_critic.pth'.format(index + 1))

    def load_model(self):
        """Load learnable model's parameters of each agent."""
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'

        for index, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('agent{}_checkpoint_actor.pth'.format(index + 1),  map_location=map_location))
            agent.critic_local.load_state_dict(torch.load('agent{}_checkpoint_critic.pth'.format(index + 1),  map_location=map_location))

