import os, sys
import time

import gymnasium as gym
import text_flappy_bird_gym

from agents.sarsa import SarsaAgent

if __name__ == "__main__":

    # initiate environment
    env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    obs, info = env.reset()

    agent = SarsaAgent()
    agent_info = {
        "num_actions": env.action_space.n,
    }
    agent.agent_init(agent_info)

    agent.load_policy("results/policy.npy")

    # iterate
    while True:

        # Select next action
        action = agent.choose_action(obs)

        print(f"Action: {action}")

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

        # Render the game
        os.system("cls")
        sys.stdout.write(env.render())
        time.sleep(0.05)  # FPS

        # If player is dead break
        if done:
            break

    env.close()
