import os, sys
import time

import gymnasium as gym
import text_flappy_bird_gym

from agents.sarsa import SarsaAgent
from agents.mc import MCControlAgent

if __name__ == "__main__":

    # initiate environment
    env = gym.make("TextFlappyBird-v0", height=15, width=20, pipe_gap=4)
    obs, info = env.reset()

    agent_type = "mc"
    agent_info = {"epsilon": 0}
    if agent_type == "sarsa":
        agent = SarsaAgent(agent_info)
    elif agent_type == "mc":
        agent = MCControlAgent(agent_info)

    # run_name = "SARSA_α_0.1_ε_0.2_ɣ_0.99_λ_1_replace"  # Infinite run!
    # run_name = "SARSA_α_0.2_ε_0.2_ɣ_0.9_λ_1_replace"
    # run_name = "SARSA_α_0.1_ε_0.1_ɣ_0.99_λ_0.5_replace"
    run_name = "MC_α_0.1_ε_0.02_ɣ_0.9"
    agent.load_policy(f"results/{run_name}.npy")

    # iterate
    iteration = 0
    while True:

        # Select next action
        action = agent.choose_action(obs)

        # Appy action and return new observation of the environment
        obs, reward, done, _, info = env.step(action)

        # Render the game
        # if iteration % 100 == 0:
        os.system("cls")
        sys.stdout.write(env.render())
        time.sleep(0.0005)  # FPS

        # If player is dead break
        if done:
            break

        iteration += 1

    env.close()
