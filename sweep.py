import os, sys
import time
import itertools

from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import text_flappy_bird_gym
from agents.sarsa import SarsaAgent


def run(alpha, epsilon, gamma, lambda_, mode, num_episodes, max_steps, env, overwrite=False):

    run_name = f"SARSA_α_{alpha}_ε_{epsilon}_ɣ_{gamma}_λ_{lambda_}_{mode}"

    if not overwrite and os.path.exists(f"results/{run_name}.npy"):
        print(f"Skipping {run_name} as it already exists")
        return [0], run_name

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
    obs, info = env.reset()

    rewards = []
    top_scores = 0
    last_top_score = 0
    last_epsilon = epsilon

    pbar = trange(num_episodes)

    for episode in pbar:
        # Change the seed for each episode
        # Each 10 steps, set the epsilon to 0 to evaluate the policy
        last_epsilon = agent.epsilon if agent.epsilon != 0 else last_epsilon
        agent.epsilon = last_epsilon if episode % 10 != 0 else 0
        agent.set_seed(episode)

        obs, _ = env.reset()
        action = agent.agent_start(obs)

        for step in range(max_steps):
            obs, reward, done, _, _ = env.step(action)
            action = agent.agent_step(reward, obs)

            if done:
                agent.agent_end(reward)
                break

        # Early stopping
        if top_scores > 10:
            break
        if episode % 10 == 0:
            if step > max_steps*0.95:
                if last_top_score == episode-10:
                    top_scores += 1
                    last_top_score = episode
                else:
                    top_scores = 0
                    last_top_score = episode
            rewards.append(step)

        pbar.set_description(f"Episode {episode + 1} - Reward: {rewards[-1]}")

    # Save the policy and the rewards
    agent.save_policy(f"results/{run_name}.npy")
    return rewards, run_name

def eval(alpha, epsilon, gamma, lambda_, mode, num_episodes, max_steps, env):

    run_name = f"SARSA_α_{alpha}_ε_{epsilon}_ɣ_{gamma}_λ_{lambda_}_{mode}"

    # initiate agent
    agent = SarsaAgent()
    agent_info = {
        "num_actions": env.action_space.n,
        "alpha": alpha,
        "epsilon": 0,
        "gamma": gamma,
        "lambda": lambda_,
        "mode": mode,
    }
    agent.agent_init(agent_info)
    agent.load_policy(f"results/{run_name}.npy")
    obs, info = env.reset()

    iteration = 0
    while iteration < max_steps:

        # Select next action
        action = agent.choose_action(obs)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

        # If player is dead break
        if done:
            break

        iteration += 1

    score = info["score"]

    return iteration, score


if __name__ == "__main__":

    # initiate environment
    env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    obs, info = env.reset()

    # Create grid search
    alphas = [0.1, 0.5, 0.2]
    epsilons = [0.3, 0.1, 0.05,  0.2]
    gammas = [0.9, 0.99]
    lambdas = [0.5, 0.9, 1]
    modes = ["replace"]#, "accumulate"]

    num_sweeps = len(alphas) * len(epsilons) * len(gammas) * len(lambdas) * len(modes)
    num_episodes = 5000
    max_steps = 200
    eval_max_steps = 1000

    rewards = {}

    print(f"Running {num_sweeps} sweeps")

    grid_search = itertools.product(alphas, epsilons, gammas, lambdas, modes)
    for sweep, (alpha, epsilon, gamma, lambda_, mode) in enumerate(grid_search):
        print(f"Running [{sweep+1}/{num_sweeps}] SARSA with alpha={alpha}, epsilon={epsilon}, gamma={gamma}, lambda={lambda_}, mode={mode}")
        reward_list, run_name = run(alpha, epsilon, gamma, lambda_, mode, num_episodes, max_steps, env, overwrite=False)
        rewards[run_name] = reward_list

        iteration, score = eval(alpha, epsilon, gamma, lambda_, mode, num_episodes, eval_max_steps, env)
        print(f"Evaluation finished after {iteration} iterations with a score of {score} {'(invicible)' if iteration == eval_max_steps else ''}")

    env.close()

    print("Sweeps finished")

    # Plot the rewards
    for run_name, reward_list in rewards.items():
        plt.plot(reward_list, label=run_name)
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()
