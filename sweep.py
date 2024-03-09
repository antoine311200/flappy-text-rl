import os, sys
import time
import itertools

from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import text_flappy_bird_gym
from agents.sarsa import SarsaAgent


def run(alpha, epsilon, gamma, lambda_, mode, num_episodes, max_steps, env):

    # initiate agent
    agent = SarsaAgent()
    agent_info = {
        "num_actions": env.action_space.n,
        "alpha": alpha,
        "epsilon": epsilon,
        "gamma": gamma,
        "lambda": lambda_,
        "mode": mode,
    }
    agent.agent_init(agent_info)

    rewards = []

    pbar = trange(num_episodes)

    for episode in pbar:
        # Change the seed for each episode
        # Each 10 steps, set the epsilon to 0 to evaluate the policy
        agent.epsilon = epsilon if episode % 10 != 0 else 0
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

    # Save the policy and the rewards
    run_name = f"SARSA_α_{alpha}_ε_{epsilon}_ɣ_{gamma}_λ_{lambda_}_{mode}"
    agent.save_policy(f"results/{run_name}.npy")
    return rewards, run_name


if __name__ == "__main__":

    # initiate environment
    env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    obs = env.reset()

    # Create grid search
    # alphas = [0.1, 0.2, 0.3]
    # epsilons = [0.005, 0.05, 0.2]
    # gammas = [0.9, 0.95, 0.99]
    # lambdas = [0.5, 0.9, 1]
    # modes = ["replace", "accumulate"]

    alphas = [0.1]
    epsilons = [0.05]
    gammas = [0.9]
    lambdas = [0.9]
    modes = ["replace"]

    num_sweeps = len(alphas) * len(epsilons) * len(gammas) * len(lambdas) * len(modes)
    num_episodes = 100
    max_steps = 100000

    rewards = {}

    print(f"Running {num_sweeps} sweeps")

    grid_search = itertools.product(alphas, epsilons, gammas, lambdas, modes)
    for sweep, (alpha, epsilon, gamma, lambda_, mode) in enumerate(grid_search):
        print(f"Running [{sweep+1}/{num_sweeps}] SARSA with alpha={alpha}, epsilon={epsilon}, gamma={gamma}, lambda={lambda_}, mode={mode}")
        reward_list, run_name = run(alpha, epsilon, gamma, lambda_, mode, num_episodes, max_steps, env)
        rewards[run_name] = reward_list

    env.close()

    print("Sweeps finished")

    # Plot the rewards
    for run_name, reward_list in rewards.items():
        plt.plot(reward_list, label=run_name)
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()
