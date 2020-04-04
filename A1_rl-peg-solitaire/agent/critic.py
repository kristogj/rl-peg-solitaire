from abc import ABC, abstractmethod
from collections import defaultdict
import random

from models import NNCritic
from torch.optim import Adam


class Critic(ABC):
    """
    An Abstract Base Class used as superclass for TableCritic and NeuralCritic
    """

    def __init__(self, config):
        self.config = config
        self.td_error = None

    @abstractmethod
    def get_value(self, state):
        pass

    def get_td_error(self, current_state, future_state, reward):
        """
        Temporal Difference (TD) error is the difference between
            a) the reward plus the discounted future state value, and
            b) the current state value.
        Intuitively, this represents the difference between
            a) the actual reward at time t+1 (r) plus the prediction of future rewards from time t+1 until the end
                of the problem-solving episode (V(s’)), and
            b) the prediction of future rewards from time t to episode end, V(s).

        The error is therefore defined as:
            td_error = reward + discount_factor * V(s') - V(s)
        where s' is the future state.
        :param current_state: str | List[int]
        :param future_state: str | List[int]
        :param reward: int
        :return: float
        """
        return reward + self.config["df_critic"] * self.get_value(future_state) - self.get_value(current_state)


class TableCritic(Critic):
    """
    A Critic using a lookup table (dict) to map states to values
    """

    def __init__(self, config):
        super(TableCritic, self).__init__(config)
        self.value_function = defaultdict(lambda: random.random())  # Value function (V) mapping states to their value
        self.eligibility = defaultdict(lambda: 1)  # Eligibility function (e)

    def get_value(self, state):
        """
        Given a state, return the estimated value of being in that state
        :param state: str
        :return: float
        """
        return self.value_function[state]

    def update_value(self, state):
        """
        Update the value that state maps to in the value function.
        The update rule is defined as:
            V(s) = V(s) + learning_rate * td_error * e(s)
        :param state: str
        :return: None
        """
        self.value_function[state] += self.config["lr_critic"] * self.td_error * self.eligibility[state]

    def update_eligibility(self, state):
        """
        Update the value that state maps to in eligibility.
        The update rule is defined as:
            e_t(s) = discount_factor * trace_decay_factor * e_{t-1}(s)
        :param state: str
        :return: None
        """
        self.eligibility[state] *= self.config["df_critic"] * self.config["tdf_critic"]

    def reset_eligibility(self):
        self.eligibility = defaultdict(lambda: 1)


class NeuralCritic(Critic):
    """
    A Critic using a neural network to map states to values
    """

    def __init__(self, config):
        super(NeuralCritic, self).__init__(config)
        self.critic = NNCritic(config["critic_layer_specs"])  # A neural network used as the value function (V)
        self.optimizer = Adam(self.critic.parameters(), lr=config["lr_critic"])

    def get_value(self, state):
        """
        Given a state, return the estimated value of being in that state
        :param state: List[int]
        :return: float
        """
        return self.critic(state)

    
