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