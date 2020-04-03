class Cell:

    def __init__(self, row, column, peg):
        # Board Location
        self.row, self.column = row, column
        self.neighbours = {}
        self.peg = peg
        pass

    def add_neighbour(self, cell):
        self.neighbours[(cell.row, cell.column)] = cell

    def __repr__(self):
        return str(self.peg)

    def __str__(self):
        return str(self.peg)
