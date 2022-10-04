import numpy as np
import random
from collections import namedtuple, deque
from copy import copy

from model import Actor, Critic

import torch
import torch.nn.functional as F
from torch import optim as optim

BUFFER_SIZE = int(1e5)   # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-3               # for soft update of target parameters
LR_ACTOR = 1e-3          # learning rate (Actor)
LR_CRITIC = 1e-4         # learning rate (Critic)
L2_WEIGHT_DECAY = 10e-2  # L2 Weight Decay from Paper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, checkpoints=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor Network
        # Specify both a local network and a target network that is
        # periodically updated by the local network (either soft update or full
        # update)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(
                self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        # Ditto to the actor network above.  Starting with same setup.
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(
                self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=L2_WEIGHT_DECAY)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Set up the noise
        self.noise = OUProcess(action_size, seed)

        if checkpoints is not None:
            actor = torch.load(checkpoints[0])
            critic = torch.load(checkpoints[1])

            # Configure both networks with the saved actor information
            self.actor_local.load_state_dict(actor)
            self.actor_target.load_state_dict(actor)

            # And also set up the critic
            self.critic_local.load_state_dict(critic)
            self.critic_target.load_state_dict(critic)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, include_noise=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            include_noise (bool): Whether or not to include noise on the samples (Ornstein-Uhlenbeck Process)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # If we are planning to add noise, go ahead and add it
        if include_noise:
            action += self.noise.get_sample()

        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Recall that the actor is attempting to predict the next action, and the critic is attempting to predict the Q-value.

        actor_target(state) returns the next action
        critic_target(state, action) returns a q value
        Q_targets = reward + gamma * critic_target(next_state, actor_target(next_state))

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## Update the critic
        next_actions = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, next_actions)

        # Compute the q targets, but not for episodes that complete
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.critic_local(states, actions)
        loss = F.mse_loss(q_expected, q_targets)

        # use the optimizer with gradient info to minimize the loss going
        # forward
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        # Use the critic update information to update the actor
        predicted_actions = self.actor_local(states)
        # Note the negative sign here.  This isn't a minimization problem, we
        # are actually trying to maximize the return value here so we need to
        # use the negative.
        loss = -self.critic_local(states, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # Now update the target networks with a soft update
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def reset(self):
        # Reset the noise function back to the original mean value
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter

        Blatant reuse from last project lol.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUProcess(object):

    """ Add noise according to Ornstein-Uhlenbeck Process """

    def __init__(self, action_size, seed, mu=0., theta=0.15, sigma=0.1):
        """Set up the noise process parameters"""
        self.mu = mu * np.ones(action_size)
        self.state = self.mu
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """ Reset the internal noise to the original mean of mu """
        # Note the call to copy here... pass by reference really makes this
        # confusing in python sometimes
        self.state = copy(self.mu)

    def get_sample(self):
        dx = self.theta * (self.mu - self.state) + \
            self.sigma * np.random.rand(self.state.shape[0])
        self.state += dx
        return self.state
