#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py.
"""

from __future__ import print_function
from abc import ABCMeta, abstractmethod
from collections import defaultdict

import numpy as np

from utils import create_epsilon_greedy_policy


class BaseAgent:
    """Implements the agent for an RL-Glue environment.
    Note:
        agent_init, agent_start, agent_step, agent_end, agent_cleanup, and
        agent_message are required methods.
    """

    __metaclass__ = ABCMeta

    def __init__(self, agent_info= {}):
        self.agent_init(agent_info)

    @abstractmethod
    def agent_init(self, agent_info= {}):
        """Setup for the agent called when the experiment first starts."""

    @abstractmethod
    def agent_start(self, observation, env):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, observation, env):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward, env):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""

    @abstractmethod
    def agent_message(self, message):
        """A function used to pass information from the agent to the experiment.
        Args:
            message: The message passed to the agent.
        Returns:
            The response (or answer) to the message.
        """

class GreedyAgent(BaseAgent):

    def agent_init(self, agent_info= {}):
        self.epsilon = agent_info.get("epsilon", 0.02)
        self.alpha = agent_info.get("alpha", 0.2)
        self.gamma = agent_info.get("gamma", 0.9)
        self.num_actions = agent_info.get("num_actions", 2)

        self.Q = defaultdict(lambda: np.zeros(self.num_actions))

        self.set_seed(agent_info.get("seed", 42))

    def set_seed(self, seed=42):
        self.rand_generator = np.random.RandomState(seed)
        self.policy = create_epsilon_greedy_policy(
            self.Q, self.epsilon, self.num_actions, self.rand_generator
        )

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