class Player:

    def __init__(self, name):
        self.name = name  # Name of the player who is playing

    @staticmethod
    def perform_action(action):
        """
        Perform move on the board such that the state of the board changes
        :param action: Action
        :return: None
        """
        action.from_.peg = False
        action.over.peg = False
        action.to_.peg = True
