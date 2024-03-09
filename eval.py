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
    obs = env.reset()

    # Create grid search
    alphas = [0.1, 0.2, 0.3]
    epsilons = [0.005, 0.05, 0.2]
    gammas = [0.9, 0.95, 0.99]
    lambdas = [0.5, 0.9, 1]
    modes = ["replace", "accumulate"]

    num_sweeps = len(alphas) * len(epsilons) * len(gammas) * len(lambdas) * len(modes)
    num_episodes = 1000
    max_steps = 1000

    iterations = {}
    scores = {}

    print(f"Running {num_sweeps} sweeps")

    grid_search = itertools.product(alphas, epsilons, gammas, lambdas, modes)
    for sweep, (alpha, epsilon, gamma, lambda_, mode) in enumerate(grid_search):
        print(f"Running [{sweep+1}/{num_sweeps}] SARSA with alpha={alpha}, epsilon={epsilon}, gamma={gamma}, lambda={lambda_}, mode={mode}")
        run_name = f"SARSA_α_{alpha}_ε_{epsilon}_ɣ_{gamma}_λ_{lambda_}_{mode}"
        iterations[run_name], scores[run_name] = run(alpha, epsilon, gamma, lambda_, mode, num_episodes, max_steps, env)
        invicible = iterations[run_name] == max_steps
        print(f"Run finished in {iterations[run_name]} steps with a score of {scores[run_name]} {'(invicible)' if invicible else ''}")

    env.close()

    print("Sweeps finished")
