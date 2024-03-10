from collections import defaultdict
import numpy as np

from agents.agent import BaseAgent, GreedyAgent
from utils import create_epsilon_greedy_policy

class SarsaAgent(GreedyAgent):

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        super().agent_init(agent_info)

        self.last_action = None
        self.last_state = None

        self.lambda_ = agent_info.get("lambda", 0.9)
        self.mode = agent_info.get("mode", "replace")

        self.E = defaultdict(lambda: np.zeros(self.num_actions))

    def agent_start(self, state):
        action = self.choose_action(state)
        self.last_state = state
        self.last_action = action

        return action

    def agent_step(self, reward, state):
        action = self.choose_action(state)

        delta = (
            reward
            + self.gamma * self.Q[state][action]
            - self.Q[self.last_state][self.last_action]
        )
        self.E[self.last_state][self.last_action] += 1

        for s, _ in self.Q.items():
            self.Q[s][:] += self.alpha * delta * self.E[s][:]
            if self.mode == "replace":
                if s == state:
                    self.E[s][:] = 1
                else:
                    self.E[s][:] *= self.gamma * self.lambda_
            elif self.mode == "accumulate":
                self.E[s][:] *= self.gamma * self.lambda_

        self.last_state = state
        self.last_action = action

        return action

    def agent_end(self, reward):
        delta = reward - self.Q[self.last_state][self.last_action]
        self.E[self.last_state][self.last_action] += 1

        for s, _ in self.Q.items():
            self.Q[s][:] += self.alpha * delta * self.E[s][:]