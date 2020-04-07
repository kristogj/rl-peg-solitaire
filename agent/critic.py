from abc import ABC, abstractmethod
from collections import defaultdict
import random

from models import NNCritic
from torch.optim import Adam
import torch


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
        self.td_error = reward + self.config["df_critic"] * self.get_value(future_state) - self.get_value(current_state)
        return self.td_error

    @abstractmethod
    def reset_eligibility(self):
        pass


class TableCritic(Critic):
    """
    A Critic using a lookup table (dict) to map states to values
    """

    def __init__(self, config):
        super(TableCritic, self).__init__(config)
        self.value_function = defaultdict(lambda: random.random())  # Value function (V) mapping states to their value
        self.eligibility = defaultdict(lambda: 0)  # Eligibility function (e)

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
        self.eligibility[state] *= self.config["df_critic"] * self.config["dr_critic"]

    def set_eligibility(self, state, value):
        """
        Set the value that state maps to in eligibility to value.
        :param state: str
        :param value: int
        :return:
        """
        self.eligibility[state] = value

    def reset_eligibility(self):
        self.eligibility = defaultdict(lambda: 0)


class NeuralCritic(Critic):
    """
    A Critic using a neural network to map states to values
    """

    def __init__(self, config):
        super(NeuralCritic, self).__init__(config)
        self.value_function = NNCritic(config["critic_layer_specs"])  # A neural network used as the value function (V)
        self.optimizer = Adam(self.value_function.parameters(), lr=config["lr_critic"])
        self.eligibility = [torch.zeros(param.shape) for param in self.value_function.parameters()]

    def get_value(self, state):
        """
        Given a state, return the estimated value of being in that state
        :param state: str
        :return: float
        """
        state = list(map(int, list(state)))
        state = torch.FloatTensor(state)
        return self.value_function(state)

    def update_value(self, state):
        with torch.no_grad():
            for p, eligibility in zip(self.value_function.parameters(), self.eligibility):
                new_val = p + self.config["lr_critic"] * self.td_error * eligibility
                p.copy_(new_val)

    def set_eligibility(self, state, value):
        self.value_function.zero_grad()
        self.td_error.backward()

    def update_eligibility(self, state):
        with torch.no_grad():
            for p, eligibility in zip(self.value_function.parameters(), self.eligibility):
                new_val = eligibility + p.grad
                eligibility.copy_(new_val)

    def reset_eligibility(self):
        self.eligibility = [torch.zeros(param.shape) for param in self.value_function.parameters()]
