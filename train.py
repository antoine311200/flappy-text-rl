import os, sys
import time

import gymnasium as gym
import numpy as np
import text_flappy_bird_gym
from tqdm import trange

from functools import partial

from agents.sarsa import SarsaAgent
from agents.mc import MCControlAgent

def preprocess_observation(observation, env_name, env=None):
    if env_name == "TextFlappyBird":
        return observation
    elif env_name == "TextFlappyBird-screen":
        # print(observation)
        # Compute the distance to the next pipe
        # That is find the first symbol 2 in the observation matrix
        # Find the index from the observation[:, 0] and observation[:, -1] if not found

        game = env.env.env._game
        closest_upcoming_pipe = min([i for i,p in enumerate([pipe['x'] - env.env.env._game.player_x for pipe in env.env.env._game.upper_pipes]) if p>=0])
        x_dist_true = env.env.env._game.upper_pipes[closest_upcoming_pipe]['x'] - env.env.env._game.player_x
        y_dist_true = env.env.env._game.player_y-env.env.env._game.upper_pipes[closest_upcoming_pipe]['y']-env.env.env._pipe_gap//2


        bird_index = np.where(observation[6, :] == 1)[0]
        if len(bird_index) > 0:
            bird_index = bird_index[0]
        else:
            bird_index = np.where(observation[5, :] == 3)[0][0]
            if bird_index == height-1:
                bird_index += 1
            elif bird_index == 0:
                bird_index -= 1

        # Get index of the first and last 0 in the pipe line
        pipe_indices = np.where(observation[:, 0] == 2)[0]
        # Filter out the pipes that are behind the bird
        pipe_indices = pipe_indices[pipe_indices >= 6]
        first_pipe_index = int(pipe_indices[0])
        first_pipe = np.where(observation[first_pipe_index, :] == 0)[0]
        first_upper_pipe = first_pipe[0]
        first_lower_pipe = first_pipe[-1]+1
        # print(6, bird_index)
        # print(first_pipe_index, first_upper_pipe)
        # print(first_pipe_index, first_lower_pipe)

        x_dist = first_pipe_index - 6
        y_dist = bird_index - first_upper_pipe - 2
        # print(x_dist, y_dist)

        if x_dist != x_dist_true or y_dist != y_dist_true:
            print()
            print("Step")
            print(observation)
            print(closest_upcoming_pipe)
            print(game.player_x, game.player_y)
            print(game.upper_pipes)
            print(game.lower_pipes)
            print(x_dist_true, y_dist_true)
            print()
            print(6, bird_index)
            print(first_pipe_index, first_upper_pipe)
            print(first_pipe_index, first_lower_pipe)
            print(x_dist, y_dist)
            print(env.env.env._game.player_y,env.env.env._game.upper_pipes[closest_upcoming_pipe]['y'],env.env.env._pipe_gap//2)
            print(bird_index, first_upper_pipe, 2)
        # if len(pipe_index) > 1:
        #     last_pipe_index = int(pipe_index[-1])
        #     last_pipe = observation[pipe_index, :]
        #     last_pipe_space = np.where(last_pipe == 0)[0]
        #     last_first_zero = last_pipe_space[0]
        #     last_last_zero = last_pipe_space[-1]
        # else:
        # last_pipe_index = -1
        # last_first_zero = -1
        # last_last_zero = -1

        # dist_y_first = bird_index - first_first_zero
        # dist_y_last = bird_index - last_first_zero if last_pipe_index != -1 else -1
        # dist_y_middle = bird_index - (first_first_zero + first_last_zero) / 2
        # # observation = (first_pipe_index - 6, dist_y_first, dist_y_last, dist_y_middle, first_pipe_index, first_first_zero, first_last_zero, last_pipe_index, last_first_zero, last_last_zero)
        observation = (x_dist, y_dist)

        return observation

def train_SARSA(agent, env, num_episodes, max_steps, verbose_intervals=10):
    rewards = []
    long_bests = 4
    last_epsilon = agent.epsilon

    env_name = env.env.env.spec.name
    transform_observation = partial(preprocess_observation, env_name=env_name)

    pbar = trange(num_episodes)
    for episode in pbar:
        # Change the seed for each episode
        # Each 10 steps, set the epsilon to 0 to evaluate the policy
        last_epsilon = agent.epsilon if agent.epsilon != 0 else last_epsilon
        agent.epsilon = last_epsilon if episode % 10 != 0 else 0
        agent.set_seed(episode)

        obs, _ = env.reset()
        action = agent.agent_start(obs, transform=transform_observation)

        for step in range(max_steps):
            obs, reward, done, _, _ = env.step(action)
            action = agent.agent_step(reward, obs, transform=transform_observation)

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

    env_name = env.env.env.spec.name
    transform_observation = partial(preprocess_observation, env_name=env_name)

    pbar = trange(num_episodes)
    for episode in pbar:
        # Change the seed for each episode
        # Each 10 steps, set the epsilon to 0 to evaluate the policy
        last_epsilon = agent.epsilon if agent.epsilon != 0 else last_epsilon
        agent.epsilon = last_epsilon if episode % 10 != 0 else 0
        agent.set_seed(episode)

        _, _ = env.reset()
        length = agent.agent_step(env, transform=transform_observation)

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
    width = 20
    height = 15
    # env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    env = gym.make("TextFlappyBird-screen-v0", height=height, width=width, pipe_gap=4)
    obs = env.reset()

    env_name = env.env.env.spec.name

    print(f"Environment: {env_name}")

    agent_type = "mc"#"sarsa"#

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
        "epsilon": 0.08,
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01,
        "max_steps": 200,
        "alpha": 0.2,
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
