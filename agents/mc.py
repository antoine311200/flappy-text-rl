from collections import defaultdict
import numpy as np

from agents.agent import BaseAgent, GreedyAgent
from utils import create_epsilon_greedy_policy

class MCControlAgent(GreedyAgent):

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        super().agent_init(agent_info)

        self.epsilon_decay = agent_info.get("epsilon_decay", 0.999)
        self.epsilon_min = agent_info.get("epsilon_min", 0.01)

        self.max_steps = agent_info.get("max_steps", 500)

    def agent_step(self, env):
        if self.epsilon != 0.0:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        episode = self.generate_episode(env)
        states, actions, rewards = zip(*episode)

        discounts = np.array([self.gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            temp_Q = self.Q[state][actions[i]]
            self.Q[state][actions[i]] = temp_Q + self.alpha * (sum(rewards[i:]*discounts[:-(1+i)]) - temp_Q)

        return len(episode)

    def agent_end(self, env):
        self.agent_step(env)
        return

    def generate_episode(self, env):
        episode = []
        state, _ = env.reset()
        iteration = 0
        while iteration < self.max_steps:
            action = self.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
            iteration += 1
        return episode