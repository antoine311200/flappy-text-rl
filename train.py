import os, sys
import time

import gymnasium as gym
import numpy as np
import text_flappy_bird_gym
from tqdm import trange

from agents.sarsa import SarsaAgent
from agents.mc import MCControlAgent

def train_SARSA(agent, env, num_episodes, max_steps, verbose_intervals=10):
    rewards = []
    long_bests = 4
    last_epsilon = agent.epsilon

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
        if episode % verbose_intervals == 0:
            rewards.append(step)
        if sum(rewards[-long_bests:]) / long_bests > 195:
            break

        pbar.set_description(f"Episode {episode + 1} - Reward: {rewards[-1]}")

    env.close()
    return rewards

def train_MC(agent, env, num_episodes, verbose_intervals=10):
    rewards = []
    long_bests = 10
    last_epsilon = agent.epsilon

    pbar = trange(num_episodes)
    for episode in pbar:
        # Change the seed for each episode
        # Each 10 steps, set the epsilon to 0 to evaluate the policy
        last_epsilon = agent.epsilon if agent.epsilon != 0 else last_epsilon
        agent.epsilon = last_epsilon if episode % 10 != 0 else 0
        agent.set_seed(episode)

        _, _ = env.reset()
        length = agent.agent_step(env)

        # Early stopping
        if episode % verbose_intervals == 0:
            rewards.append(length)
        if sum(rewards[-long_bests:]) / long_bests > 195:
            break

        pbar.set_description(f"Episode {episode + 1} - Reward: {rewards[-1]}")

    env.close()
    return rewards

if __name__ == "__main__":

    # initiate environment
    env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    obs = env.reset()

    agent_type = "mc"#"sarsa"

    # initiate agent
    sarsa_agent_info = {
        "num_actions": env.action_space.n,
        "alpha": 0.1,
        "epsilon": 0.1,
        "gamma": 0.9,
        "lambda": 0.9,
        "mode": "replace",
    }
    mc_agent_info = {
        "num_actions": env.action_space.n,
        "epsilon": 0.02,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01,
        "max_steps": 200,
        "alpha": 0.1,
        "gamma": 0.9,
    }


    num_episodes = 50000
    max_steps = 1000
    if agent_type == "sarsa":
        agent = SarsaAgent(sarsa_agent_info)
        rewards = train_SARSA(agent, env, num_episodes, max_steps, verbose_intervals=3)
        run_name = f"SARSA_α_{sarsa_agent_info['alpha']}_ε_{sarsa_agent_info['epsilon']}_ɣ_{sarsa_agent_info['gamma']}_λ_{sarsa_agent_info['lambda']}_{sarsa_agent_info['mode']}"
    elif agent_type == "mc":
        agent = MCControlAgent(mc_agent_info)
        rewards = train_MC(agent, env, num_episodes, verbose_intervals=3)
        run_name = f"MC_α_{mc_agent_info['alpha']}_ε_{mc_agent_info['epsilon']}_ɣ_{mc_agent_info['gamma']}"

    print(f"Training finished. Run name: {run_name}")
    agent.save_policy(f"results/{run_name}.npy")


    # Plot the rewards
    import matplotlib.pyplot as plt

    plt.plot(rewards)
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()

    # Save the policy and the rewards
    agent.save_policy("results/policy.npy")
