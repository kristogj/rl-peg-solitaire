from collections import defaultdict
from utils import is_empty
import random


class Actor:

    def __init__(self, player, config):
        # Maps state-action pairs (s,a) to values that indicate the desirability of performing action a
        # when in state s
        self.config = config
        self.policy = defaultdict(float)
        self.eligibility = defaultdict(lambda: 0)
        self.player = player
        self.td_error = None
        self.epsilon = config["epsilon"]  # Epsilon for epsilon-greedy strategy
        self.dr_epsilon = config["dr_epsilon"]  # Decay rate epsilon

    def reset_eligibility(self):
        """
        Reset eligibility to default
        :return: None
        """
        self.eligibility = defaultdict(lambda: 0)

    def get_desirability(self, state, action):
        """
        Get the desirability for doing an action in a state
        :param state: str
        :param action: Action
        :return: float
        """
        return self.policy[(state, action.__str__())]

    def get_action(self, state):
        """
        Given a state of the board, return the appropriate action
        :param state: str
        :return: Action
        """
        legal_actions = list(map(lambda action: (action, self.get_desirability(state, action)),
                                 self.player.get_legal_actions()))
        if is_empty(legal_actions):
            return None
        if random.random() < self.epsilon:
            return random.choice(legal_actions)[0]
        else:
            legal_actions.sort(key=lambda tup: tup[1], reverse=True)
            return legal_actions[0][0]

    def set_eligibility(self, state, action, value):
        """
        Set the value that (state, action) maps to in eligibility to value.
        :param state: str
        :param action: Action
        :param value: int
        :return: None
        """
        self.eligibility[(state, action.__str__())] = value

    def set_td_error(self, error):
        """
        Set the td_error to error
        :param error: int
        :return: None
        """
        self.td_error = error

    def set_epsilon(self, value):
        """
        Set the value of epsilon to value
        :param: value: float
        :return: None
        """
        self.epsilon = value

    def update_policy(self, state, action):
        """
        Update the value that (state, action) maps to in the policy function.
        Update rule is defined as:
            pi(state, action) = pi(state, action) + learning_rate * td_error * e(state, action)
        :param state: str
        :param action: Action
        :return: None
        """
        self.policy[(state, action.__str__())] += self.config["lr_actor"] * self.td_error * self.eligibility[
            (state, action.__str__())]

    def update_eligibility(self, state, action):
        """
        Update the value that (state, action) maps to in eligibility.
        The update rule is defined as:
            e_t(state, action) = discount_factor * trace_factor * e_{t-1}(state, action)
        :param state: str
        :param action: Action
        :return: None
        """
        self.eligibility[(state, action.__str__())] *= self.config["df_actor"] * self.config["dr_actor"]

    def update_epsilon(self):
        """
        Update the value of epsilon.
        The update rule is defined as:
            epsilon = dr_epsilon * epsilon
        :return: None
        """
        self.epsilon = self.dr_epsilon * self.epsilon
