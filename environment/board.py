from abc import ABC, abstractmethod
from cell import Cell


class PegBoard(ABC):

    def __init__(self, config):
        self.size = config["size"]  # Size of the board
        self.type = config["type"]  # Board type. Diamond or Triangle
        self.holes_loc = config["holes_loc"]  # List of locations of all cells that should be init as empty
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]  # The actual board used while playing

    @abstractmethod
    def init_board(self):
        pass

    def init_holes(self):
        """
        Set peg attribute of Cell to False if it should be removed from the board.
        :return:
        """
        for row, column in self.holes_loc:
            try:
                self.board[row][column].is_peg = False
            except AttributeError:
                raise AttributeError("Cell({}{}) does not have 'is_peg' attribute.".format(row, column))

    def get_cell(self, coord):
        """
        Return cell at the given coordinate on the board
        :param coord: tuple (row, col)
        :return: Cell / None
        """
        row, col = coord
        return self.board[row][col]

    def get_cells(self):
        """
        Return a list of all Cells that are not None on the board
        :return: List[Cell]
        """
        return [cell for row in self.board for cell in row if cell]

    def get_empty_cells(self):
        """
        Return a list of all empty cells
        :return: List[Cell]
        """
        return [cell for cell in self.get_cells() if not cell.is_peg]

    def to_binary_string_encoding(self):
        """
        Return a binary encoding of the board as a string where 1 is_peg and 0 is empty
        :return: str
        """
        return "".join(map(lambda cell: str(int(cell.is_peg)), self.get_cells()))

    def to_binary_list_encoding(self):
        """
        Return a binary encoding of the board as a list where 1 is_peg and 0 is empty
        :return:
        """
        return list(map(lambda cell: int(cell.is_peg), self.get_cells()))

    def set_cell(self, cell):
        """
        Update board at cell position
        :param cell: Cell
        """
        self.board[cell.row][cell.column] = cell

    def set_neighbours(self, neighbour_pattern):
        """
        For each cell on the board, calculate its neighbours positions and add to list of neighbours if it is a legal
        neighbour.
        :param neighbour_pattern:
        :return: None
        """
        for r in range(self.size):
            for c in range(self.size):
                current_cell = self.get_cell((r, c))
                if current_cell:
                    neighbours = list(map(lambda tup: (r + tup[0], c + tup[1]), neighbour_pattern))
                    for i, coord in enumerate(neighbours):
                        if self.is_legal_neighbour(coord):
                            cell = self.get_cell(coord)
                            if cell:
                                current_cell.add_neighbour(cell, neighbour_pattern[i])

    def is_legal_neighbour(self, coord):
        """
        Check if coord gives position to a legal cell on the board
        :param coord:
        :return:
        """
        r, c = coord
        if (0 <= r < self.size) and (0 <= c < self.size):
            if self.get_cell((r, c)):
                return True
        else:
            return False

    def num_pegs_on_board(self):
        """
        Return number of pegs left on the board
        :return: int
        """
        return len(list(filter(lambda cell: cell.is_peg, self.get_cells())))

    def __str__(self):
        res = ""
        for row in self.board:
            res += str(row) + "\n"
        return res


class DiamondPegBoard(PegBoard):

    def __init__(self, config):
        super(DiamondPegBoard, self).__init__(config)
        # Fill board with pegs
        self.init_board()

        # Remove pegs where there should be holes
        self.init_holes()

        # Each cell will have a maximum of 6 neighbours:
        # (r-1,c) (r-1,c+1), (r,c-1), (r.c+1), (r+1,c-1), (r+1,c)
        self.pattern = [(-1, 0), (-1, 1), (0, - 1), (0, 1), (1, - 1), (1, 0)]
        self.set_neighbours(self.pattern)

    def init_board(self):
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
        self.init_board()

        # Remove pegs where there should be holes
        self.init_holes()

        # Each cell will have a maximum of 6 neighbours:
        # (r-1,c-1), (r-1,c), (r,c-1), (r, c+1), (r+1,c), (r+1, c+1)
        self.pattern = [(- 1, - 1), (- 1, 0), (0, - 1), (0, + 1), (1, 0), (1, 1)]
        self.set_neighbours(self.pattern)

    def init_board(self):
        """
        Update board with new Cell instances
        """
        for row in range(self.size):
            for column in range(row + 1):
                self.set_cell(Cell(row, column, True))
