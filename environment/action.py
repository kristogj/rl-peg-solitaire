class Action:

    def __init__(self, from_, over, to_):
        """
        Simple class to represent an action in the game
        :param from_: Cell
        :param over: Cell
        :param to_: Cell
        """
        self.from_ = from_  # Cell to move a peg from
        self.over = over  # Cell that is being moved over
        self.to_ = to_  # Cell that is being moved to

    def __repr__(self):
        return "{}-{}".format(self.from_, self.to_)
