# Deep Deterministic Policy Gradient (DDPG) approach to move a double-jointed
# arm (Project Option 1)
# ---

import torch
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt
from ddpg import Agent


# First configure the environment
# NOTE: I have configured for LINUX x86_64
# Hey UDACITY, PLEASE COMPILE A HEADLESS OPTION
env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Grab the environment info as well to set up the agent
env_info = env.reset(train_mode=True)[brain_name]

# Set up the agent generically for state and action sizes.  Can't ever be TOO
# portable!
agent = Agent(state_size=len(env_info.vector_observations[0]),
              action_size=brain.vector_action_space_size, seed=0)


# Train the agent
def ddpg(n_episodes=500, max_t=1000):
    """ Deep Deterministic Policy Gradient (DDPG)

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1, n_episodes+1):
        # Reset the environment in training mode according to the standard
        # brain
        env_info = env.reset(train_mode=True)[brain_name]

        # Get the initial state from the environment
        state = env_info.vector_observations[0]

        # Initialize the score to zero
        score = 0
        # We will leave a maximum time in here for now
        for _ in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        if i_episode % 10 == 0:
            print(
                    '\rEpisode {}\tAverage Score: {:.2f}'.format(
                        i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= 30:
            print(
                    '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                        i_episode-100, np.mean(scores_window)))
            # If we win, we need to save the checkpoint
            torch.save(agent.actor_local.state_dict(), 'checkpoint-actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint-critic.pth')
            break

    # Close the environment
    env.close()
    return scores


# Train the agent and output the scores per episode
score_array = ddpg()

# Plot the scores over each episode
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(score_array)), score_array)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
