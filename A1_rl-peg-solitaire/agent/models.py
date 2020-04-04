from torch import nn


class NNCritic(nn.Module):
    """
    The Neural Network model used by the NeuralCritic to map states to values
    """

    def __init__(self, layer_specs):
        super(NNCritic, self).__init__()
        self.model = nn.Sequential()

        for x in range(1, len(layer_specs)):
            layer = nn.Linear(in_features=layer_specs[x - 1], out_features=layer_specs[x])
            self.model.add_module("Layer {}".format(x), layer)

    def forward(self, encoded_board):
        """
        Forward propagate the binary encoded board
        :param encoded_board: List[int]
        :return:
        """
        return self.model(encoded_board)
