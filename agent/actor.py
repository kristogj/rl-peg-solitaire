from collections import defaultdict
from utils import is_empty
import random
import logging


class Actor:

    def __init__(self, player, config):
        self.config = config
        self.policy = defaultdict(lambda: 0)  # Policy mapping SAP to its desirability value
        self.eligibility = defaultdict(lambda: 0)
        self.player = player  # Player who performs all the actions
        self.td_error = None  # Temporal Difference error
        self.epsilon = config["epsilon"]  # Epsilon for epsilon-greedy strategy

        # Stats
        self.random_actions = 0
        self.total_actions = 0
        self.last_epsilon = None

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
        self.total_actions += 1
        if random.random() < self.epsilon:
            self.random_actions += 1
            return random.choice(legal_actions)[0]
        else:
            legal_actions.sort(key=lambda tup: tup[1], reverse=True)
            return legal_actions[0][0]

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
        self.last_epsilon = self.epsilon
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

    def update_eligibility(self, state, action, is_current_state=False):
        """
        Update the value that (state, action) maps to in eligibility.
        The update rule is defined as:
            e_t(state, action) = discount_factor * trace_factor * e_{t-1}(state, action)
        :param state: str
        :param action: Action
        :param is_current_state: boolean
        :return: None
        """
        if is_current_state:
            self.eligibility[(state, action.__str__())] = 1
        else:
            self.eligibility[(state, action.__str__())] *= self.config["df_actor"] * self.config["dr_actor"]

    def update_epsilon(self):
        """
        Update the value of epsilon.
        The update rule is defined as:
            epsilon = dr_epsilon * epsilon
        :return: None
        """
        self.epsilon *= self.config["dr_epsilon"]

    def report_actor_stats(self):
        """
        Log general stats of what the Actor has done during its existence
        """
        avg_policy = sum(self.policy.values()) / len(self.policy.keys())
        avg_eligibility = sum(self.eligibility.values()) / len(self.eligibility.keys())
        p = int(100 * self.random_actions / self.total_actions)
        logging.info("ACTOR: ")
        logging.info("\t {} of {} actions where random: {}%".format(self.random_actions, self.total_actions, p))
        logging.info("\t Epsilon ended at: {}".format(self.last_epsilon))
        logging.info("\t Avg policy values: {}".format(avg_policy if isinstance(avg_policy, float) else avg_policy.item()))
        logging.info("\t Avg eligibility values: {}".format(avg_eligibility))
