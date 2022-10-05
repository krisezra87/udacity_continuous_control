# Deep Deterministic Policy Gradient (DDPG) approach to move a double-jointed
# arm (Project Option 1)
# ---

import torch
from unityagents import UnityEnvironment
from ddpg import Agent


# First configure the environment
# NOTE: I have configured for LINUX x86_64
env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Set up the agent generically for state and action sizes.  Can't ever be TOO
# portable!
agent = Agent(state_size=brain.vector_observation_space_size,
              action_size=brain.vector_action_space_size, seed=0)

# Load the successful networks
actor = torch.load('checkpoint-actor.pth')
critic = torch.load('checkpoint-critic.pth')

# Configure both networks with the saved actor/critic information
agent.actor_local.load_state_dict(actor)
agent.actor_target.load_state_dict(actor)

# And also set up the critic
agent.critic_local.load_state_dict(critic)
agent.critic_target.load_state_dict(critic)

# demo the agent
def ddpg_demo(max_t=1200):
    """ Deep Deterministic Policy Gradient (DDPG)

    Params
    ======
        max_t (int): maximum number of timesteps per episode
    """
    scores = []

    env_info = env.reset(train_mode=False)[brain_name]

    # Get the initial state from the environment
    state = env_info.vector_observations[0]

    # Get the agent ready for this new episode
    agent.reset()

    # Initialize the score to zero
    score = 0
    # We will leave a maximum time in here for now
    for t_step in range(max_t):
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
    scores.append(score)              # save most recent score

    # Close the environment
    env.close()
    return scores


# Do a run and output the score
score = ddpg_demo()
print("Score: {}".format(score))
