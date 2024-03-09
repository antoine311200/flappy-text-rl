import os, sys
import time

import gymnasium as gym
import numpy as np
import text_flappy_bird_gym
from tqdm import trange

from agents.sarsa import SarsaAgent


if __name__ == "__main__":

    # initiate environment
    env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    obs = env.reset()

    # initiate agent
    agent = SarsaAgent()
    agent_info = {
        "num_actions": env.action_space.n,
    }
    agent.agent_init(agent_info)

    init_epsilon = agent.epsilon
    num_episodes = 1000
    max_steps = 100000

    rewards = []

    pbar = trange(num_episodes)

    for episode in pbar:
        # Change the seed for each episode
        # Each 10 steps, set the epsilon to 0 to evaluate the policy
        agent.epsilon = init_epsilon if episode % 10 != 0 else 0
        agent.set_seed(episode)

        obs, info = env.reset()
        action = agent.agent_start(obs)

        for step in range(max_steps):
            obs, reward, done, _, info = env.step(action)
            action = agent.agent_step(reward, obs)

            if done:
                agent.agent_end(reward)
                break

        if episode % 10 == 0:
            rewards.append(step)

        pbar.set_description(f"Episode {episode + 1} - Reward: {rewards[-1]}")

    env.close()

    # Plot the rewards
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()

    # Save the policy and the rewards
    agent.save_policy("results/policy.npy")