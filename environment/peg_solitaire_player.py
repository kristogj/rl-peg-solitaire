from action import Action


class Player:

    def __init__(self, board, config):
        self.config = config
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

    def get_legal_actions(self):
        """
        Return a list of all legal actions that can be done on the board.
        :return: List[Action]
        """
        legal_actions = []
        for empty_cell in self.board.get_empty_cells():
            # Check neighbours of all empty cells
            neighbours = empty_cell.get_neighbours().values()
            for neighbour in neighbours:
                neighbour_cell = neighbour["cell"]
                # If the neighbour is a peg, you should calculate the neighbour that could do a legal move over it
                if neighbour_cell.peg:
                    pattern = neighbour["pattern"]
                    coord = (neighbour_cell.row + pattern[0], neighbour_cell.column + pattern[1])
                    if self.board.is_legal_neighbour(coord) and self.board.get_cell(coord).peg:
                        legal_actions.append(Action(self.board.get_cell(coord), neighbour_cell, empty_cell))
        return legal_actions
