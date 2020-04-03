from abc import ABC, abstractmethod
from cell import Cell


class PegBoard(ABC):

    def __init__(self, config):
        print(config)
        self.size = config["size"]
        self.type = config["type"]
        self.holes_loc = config["holes_loc"]
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]

    @abstractmethod
    def fill_board(self):
        pass

    def remove_pegs(self):
        """
        Set peg attribute of Cell to False if it should be removed from the board.
        :return:
        """
        for row, column in self.holes_loc:
            try:
                self.board[row][column].peg = False
            except AttributeError:
                raise AttributeError("Cell({}{}) does not have 'peg' attribute.".format(row, column))

    def get_cell(self, coord):
        """
        Return cell at the given coordinate on the board
        :param coord: tuple (row, col)
        :return: Cell / None
        """
        row, col = coord
        return self.board[row][col]

    def set_cell(self, cell):
        """
        Update the board at the cell position to be the cell instance.
        :param cell: Cell
        """
        self.board[cell.row][cell.column] = cell

    def is_legal_neighbour(self, coord):
        r, c = coord
        return (0 <= r < self.size) and (0 <= c < self.size)

    def set_neighbours(self, neighbour_pattern):
        for r in range(self.size):
            for c in range(self.size):
                current_cell = self.get_cell((r, c))
                if current_cell:
                    neighbours = list(map(lambda tup: (r + tup[0], c + tup[1]), neighbour_pattern))
                    for coord in neighbours:
                        if self.is_legal_neighbour(coord):
                            cell = self.get_cell(coord)
                            if cell:
                                current_cell.add_neighbour(cell)

    def __str__(self):
        res = ""
        for row in self.board:
            res += str(row) + "\n"
        return res


class DiamondPegBoard(PegBoard):

    def __init__(self, config):
        super(DiamondPegBoard, self).__init__(config)
        # Fill board with pegs
        self.fill_board()

        # Remove pegs where there should be holes
        self.remove_pegs()

        # Each cell will have a maximum of 6 neighbours:
        # (r-1,c) (r-1,c+1), (r,c-1), (r.c+1), (r+1,c-1), (r+1,c)
        self.set_neighbours([(-1, 0), (-1, 1), (0, - 1), (0, 1), (1, - 1), (1, 0)])

    def fill_board(self):
        """
        Update board with new Cell instances
        """
        for row in range(self.size):
            for column in range(self.size):
                self.set_cell(Cell(row, column, True))


class TrianglePegBoard(PegBoard):

    def __init__(self, config):
        super(TrianglePegBoard, self).__init__(config)
        # Fill board with pegs
        self.fill_board()

        # Remove pegs where there should be holes
        self.remove_pegs()

        # Each cell will have a maximum of 6 neighbours:
        # (r-1,c-1), (r-1,c), (r,c-1), (r, c+1), (r+1,c), (r+1, c+1)
        self.set_neighbours([(- 1, - 1), (- 1, 0), (0, - 1), (0, + 1), (1, 0), (1, 1)])

    def fill_board(self):
        """
        Update board with new Cell instances
        """
        for row in range(self.size):
            for column in range(row + 1):
                self.set_cell(Cell(row, column, True))


def get_board(config):
    if config["type"] == "d":
        return DiamondPegBoard(config)
    elif config["type"] == "t":
        return TrianglePegBoard(config)
