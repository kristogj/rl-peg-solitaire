from abc import ABC, abstractmethod
from collections import defaultdict
import random
import logging

from models import NNCritic
from torch.optim import Adam
import torch


class Critic(ABC):
    """
    An Abstract Base Class used as superclass for TableCritic and NeuralCritic
    """

    def __init__(self, config):
        self.config = config
        self.td_error = None  # Temporal Difference error

        # Stats
        self.td_errors = []

    @abstractmethod
    def get_value(self, state):
        pass

    def get_td_error(self, current_state, future_state, reward):
        """
        Temporal Difference (TD) error is the difference between the reward plus the discounted future state value,
        and the current state value.
        The error is therefore defined as:
            td_error = reward + discount_factor * V(s') - V(s)
        where s' is the future state.
        :param current_state: str | List[int]
        :param future_state: str | List[int]
        :param reward: int
        :return: float
        """
        self.td_error = reward + self.config["df_critic"] * self.get_value(future_state) - self.get_value(current_state)
        self.td_errors.append(self.td_error)
        return self.td_error

    @abstractmethod
    def reset_eligibility(self):
        pass

    @abstractmethod
    def report_critic_stats(self):
        pass


class TableCritic(Critic):
    """
    A Critic using a lookup table (dict) to map states to values
    """

    def __init__(self, config):
        super(TableCritic, self).__init__(config)
        self.value_function = defaultdict(lambda: random.random() * 0.2)  # Value function (V)
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

    def update_eligibility(self, state, is_current_state=False):
        """
        Update the value that state maps to in eligibility.
        The update rule is defined as:
        if state is current state:
            e_t(s) = 1
        else:
            e_t(s) = discount_factor * trace_decay_factor * e_{t-1}(s)
        :param state: str
        :param is_current_state: boolean
        :return: None
        """
        if is_current_state:
            self.eligibility[state] = 1
        else:
            self.eligibility[state] *= self.config["df_critic"] * self.config["dr_critic"]

    def reset_eligibility(self):
        """
        Reset eligibility table back to default
        """
        self.eligibility = defaultdict(lambda: 0)

    def report_critic_stats(self):
        """
        Log general stats about what the critic has calculated during its existence
        """
        avg_value = sum(self.value_function.values()) / len(self.value_function.keys())
        avg_eligibility = sum(self.eligibility.values()) / len(self.eligibility.keys())
        avg_td_error = sum(self.td_errors) / len(self.td_errors)
        logging.info("Critic: ")
        logging.info("\t Total value mapping: {}".format(len(self.value_function.keys())))
        logging.info("\t Avg value function values: {}".format(avg_value))
        logging.info("\t Total eligibility values: {}".format(len(self.eligibility.keys())))
        logging.info("\t Avg eligibility values: {}".format(avg_eligibility))
        logging.info("\t Avg TD errors: {}".format(avg_td_error))


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
        value = self.value_function(state)
        return value

    def update_value(self, state):
        """
        with torch.no_grad():
            for p, eligibility in zip(self.value_function.parameters(), self.eligibility):
                new_val = p + self.config["lr_critic"] * self.td_error * eligibility
                p.copy_(new_val)

        with torch.no_grad():
            for i, weight in enumerate(self.value_function.parameters()):

                eligibility_trace = self.config["df_critic"] * self.config["lr_critic"] * self.eligibility[i]
                torch.add(weight, eligibility_trace)
        """
        self.optimizer.zero_grad()
        outputs = self.get_value(state)
        outputs.backward()
        for i, weight in enumerate(self.value_function.parameters()):
            self.eligibility[i] += weight.grad
            weight.grad = -self.td_error * self.eligibility[i]
        self.optimizer.step()

    def update_eligibility(self, state):
        # with torch.no_grad():
        # for p, eligibility in zip(self.value_function.parameters(), self.eligibility):
        # new_val = eligibility + p.grad
        # eligibility.copy_(new_val)
        pass

    def reset_eligibility(self):
        self.eligibility = [torch.zeros(param.shape) for param in self.value_function.parameters()]

    def report_critic_stats(self):
        pass
