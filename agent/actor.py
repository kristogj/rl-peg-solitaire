"""
POLICY:
The actor’s policy should be represented as a table (or Python dictionary) that maps state-action pairs (s,a)
to values that indicate the desirability of performing action a when in state s. For any state s, it is wise to
normalize the values across all legal actions from s, thus yielding a probability distribution over the possible
actions to take from s. The  greedy policy would then choose the action with the highest probability, or a
random value.

EPSILON:
To insure a balance between exploration and exploitation, the actor should use its policy in an -greedy
manner (see actor-critic.pdf), where eps is either a constant, user-supplied parameter, or a dynamic variable
that changes (i.e. decreases) from earlier to later episodes. By setting eps = 0 at run’s end, the behavior policy
essentially becomes the target policy. By displaying one game played with this policy, the user sees the best
moves that the actor has found for the states of an episode.
"""
from collections import defaultdict


class Actor:

    def __init__(self, player, config):
        # Maps state-action pairs (s,a) to values that indicate the desirability of performing action a
        # when in state s
        self.config = config
        self.policy = defaultdict(float)
        self.eligibility = defaultdict(lambda: 0)
        self.player = player
        self.td_error = None

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
        return self.policy[(state, action)]

    def get_action(self, state):
        """
        Given a state of the board, return the appropriate action
        :param state: str
        :return: Action
        """
        legal_actions = list(map(lambda action: (action, self.get_desirability(state, action)),
                                 self.player.get_legal_actions()))
        legal_actions.sort(key=lambda tup: tup[1], reverse=True)
        try:
            return legal_actions[0][0]
        except IndexError:
            return None

    def set_eligibility(self, state, action, value):
        """
        Set the value that (state, action) maps to in eligibility to value.
        :param state: str
        :param action: Action
        :param value: int
        :return: None
        """
        self.eligibility[(state, action)] = value

    def set_td_error(self, error):
        """
        Set the td_error to error
        :param error: int
        :return: None
        """
        self.td_error = error

    def update_policy(self, state, action):
        """
        Update the value that (state, action) maps to in the policy function.
        Update rule is defined as:
            pi(state, action) = pi(state, action) + learning_rate * td_error * e(state, action)
        :param state: str
        :param action: Action
        :return: None
        """
        self.policy[(state, action)] += self.config["lr_actor"] * self.td_error * self.eligibility[(state, action)]

    def update_eligibility(self, state, action):
        """
        Update the value that (state, action) maps to in eligibility.
        The update rule is defined as:
            e_t(state, action) = discount_factor * trace_factor * e_{t-1}(state, action)
        :param state: str
        :param action: Action
        :return: None
        """
        self.eligibility[(state, action)] *= self.config["df_actor"] * self.config["tdf_actor"]
