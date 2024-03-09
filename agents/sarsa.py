from agents.agent import BaseAgent

from collections import defaultdict
import numpy as np


def argmax(q_values, rand_generator):
    """argmax with random tie-breaking
    Args:
        q_values (Numpy array): the array of action-values
    Returns:
        action (int): an action with the highest value
    """
    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return rand_generator.choice(ties)


def create_epsilon_greedy_policy(Q, epsilon, num_actions, rand_generator):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
    Q: A dictionary that maps from state -> action-values.
        Each value is a numpy array of length num_actions (see below)
    epsilon: The probability to select a random action . float between 0 and 1.
    num_actions: Number of actions in the environment.

    Returns:
    A function that takes the observation as an argument and returns
    the probabilities for each action in the form of a numpy array of length num_actions.
    """

    def policy_fn(observation):
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = argmax(Q[observation], rand_generator)
        A[best_action] += 1.0 - epsilon
        return A

    return policy_fn


class SarsaAgent(BaseAgent):

    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts."""
        self.last_action = None
        self.last_state = None
        self.epsilon = agent_info.get("epsilon", 0.02)
        self.alpha = agent_info.get("alpha", 0.2)
        self.gamma = agent_info.get("gamma", 0.9)
        self.num_actions = agent_info.get("num_actions", 2)
        self.lambda_ = agent_info.get("lambda", 0.9)
        self.mode = agent_info.get("mode", "replace")

        self.Q = defaultdict(lambda: np.zeros(self.num_actions))
        self.E = defaultdict(lambda: np.zeros(self.num_actions))

        self.set_seed(agent_info.get("seed", 42))

    def set_seed(self, seed=42):
        self.rand_generator = np.random.RandomState(seed)
        self.policy = create_epsilon_greedy_policy(
            self.Q, self.epsilon, self.num_actions, self.rand_generator
        )

    def agent_start(self, state):
        # print("Initial state: ", state)
        action = self.choose_action(state)
        # print("Action taken:", action)
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

    def choose_action(self, state):
        """returns the action that the agent chooses at a particular state using an epsilon-soft policy"""
        # print self.Q as dictionary
        action_probs = self.policy(state)
        action = self.rand_generator.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    def save_policy(self, file_path):
        """Saves the Q-value function defaultdict"""
        np.save(file_path, dict(self.Q))

    def load_policy(self, file_path):
        """Loads the Q-value function defaultdict"""
        self.Q = defaultdict(lambda: np.zeros(self.num_actions), np.load(file_path, allow_pickle=True).item())
        self.policy = create_epsilon_greedy_policy(
            self.Q, self.epsilon, self.num_actions, self.rand_generator
        )