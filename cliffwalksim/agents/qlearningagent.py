import gymnasium as gym
import numpy as np
from cliffwalksim.agents.tabularagent import TabularAgent


class QLearningAgent(TabularAgent):
    """
    Q-learning agent implementation.
    """
    def __init__(self, state_space: gym.spaces.Discrete, action_space: gym.spaces.Discrete, learning_rate=0.1,
                 discount_rate=0.9, epsilon=0.1):
        """
        Q-learning agent constructor.
        :param state_space: state space of gym environment.
        :param action_space: action space of gym environment.
        :param learning_rate: learning rate.
        :param discount_rate: discount rate.
        :param epsilon: epsilon for epsilon-greedy policy.
        """
        super().__init__(state_space, action_space, learning_rate, discount_rate)
        self.epsilon = epsilon

    def update(self, trajectory: tuple) -> None:
        """
        Update Q-values using Q-learning algorithm.
        :param trajectory: (state, action, reward, next_state).
        """
        state, action, reward, next_state = trajectory
        old_q_value = self.q_table[state, action]
        max_next_q_value = np.max(self.q_table[next_state])
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_rate * max_next_q_value - old_q_value)
        self.q_table[state, action] = new_q_value

    def policy(self, state):
        """
        Epsilon-greedy policy.
        :param state: current state.
        :return: action.
        """
        if np.random.rand() < self.epsilon:
            return self.env_action_space.sample()  # Random action
        else:
            return np.argmax(self.q_table[state])  # Greedy action

