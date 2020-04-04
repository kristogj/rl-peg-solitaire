class Player:

    def __init__(self, name, board):
        self.name = name  # Name of the player who is playing
        self.board = board  # The board the game is played on

    def perform_action(self, action):
        """
        Perform the action on the board and return the new state.
        :param action: Action
        :return:
        """
        action.from_.peg = False
        action.over.peg = False
        action.to_.peg = True

        return self.board.to_binary_string_encoding()
