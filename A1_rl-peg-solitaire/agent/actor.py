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


class Actor:

    def __init__(self, config):
        # Maps state-action pairs (s,a) to values that indicate the desirability of performing action a
        # when in state s
        self.policy = {}

        self.config = config

    def get_action(self, state):
        """
        Given a state of the board, return the appropriate action
        :param state: str
        :return: Action
        """
        return self.policy[state]

    def update_state(self, state, action):
        """
        Update the action that state map to in the policy
        :param state: str
        :param action: Action
        :return: None
        """
        self.policy[state] = action
