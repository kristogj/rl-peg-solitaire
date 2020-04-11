from torch import nn


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class NNCritic(nn.Module):
    """
    The Neural Network model used by the NeuralCritic to map states to values
    """

    def __init__(self, layer_specs):
        super(NNCritic, self).__init__()
        self.model = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

        for x in range(1, len(layer_specs)):
            layer = nn.Linear(in_features=layer_specs[x - 1], out_features=layer_specs[x], bias=False)
            self.model.add_module("Layer {}".format(x), layer)

        self.model.apply(init_weights)

    def forward(self, encoded_board):
        """
        Forward propagate the binary encoded board
        :param encoded_board: List[int]
        :return:
        """
        x = encoded_board
        for i in range(len(self.model)):
            x = self.relu(self.model[i](x))
        return x
