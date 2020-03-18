from abc import ABC, abstractmethod


class PegBoard(ABC):

    def __init__(self, config):
        print(config)
        self.config = config
        self.board = [[None for _ in range(config["size"])] for _ in range(config["size"])]

    @abstractmethod
    def fill_board(self):
        pass

    def remove_pegs(self):
        # Remove pegs to add holes to the board
        for row, column in self.config["holes_loc"]:
            try:
                self.board[row][column].peg = False
            except AttributeError:
                raise AttributeError("Cell({}{}) does not have 'peg' attribute.".format(row, column))


class DiamondPegBoard(PegBoard):

    def __init__(self, config):
        super(DiamondPegBoard, self).__init__(config)
        self.fill_board()

    def fill_board(self):
        size = self.config["size"]

        for row in range(size):
            for column in range(size):
                self.board[row][column] = Cell(row, column, True)

        # Add holes to the board
        self.remove_pegs()


class TrianglePegBoard(PegBoard):

    def __init__(self, config):
        super(TrianglePegBoard, self).__init__(config)
        self.fill_board()

    def fill_board(self):
        size = self.config["size"]

        for row in range(size):
            for column in range(row + 1):
                self.board[row][column] = Cell(row, column, True)

        # Add holes to the board
        self.remove_pegs()


class Cell:

    def __init__(self, row, column, peg):
        # Board Location
        self.row, self.column = row, column
        self.neighbours = None
        self.peg = peg
        pass

    def __repr__(self):
        return str(self.peg)

    def __str__(self):
        return str(self.peg)


def get_board(config):
    if config["type"] == "d":
        return DiamondPegBoard(config)
    elif config["type"] == "t":
        return TrianglePegBoard(config)
