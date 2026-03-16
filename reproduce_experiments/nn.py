from torch import nn

class LaueNN(nn.Module):
    def __init__(self, input_size, output_size, dropout, *args):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_size, (output_size*15 + input_size)//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear((output_size*15 + input_size)//2, output_size*15),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_size*15, output_size),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


class CustomNN(nn.Module):
    def __init__(self, input_size, output_size, input_dropout = 0., dims = None,
                 activation = 'relu', **kwargs):
        super().__init__()

        if activation == 'relu':
            act_fnc = nn.ReLU()
        elif activation == 'sigmoid':
            act_fnc = nn.Sigmoid()
        elif activation == 'tanh':
            act_fnc = nn.Tanh()
        else:
            print("Activation must be 'relu', 'sigmoid' or 'tanh'. Defaulting to 'relu'.")
            act_fnc = nn.ReLU()

        modules = []

        if input_dropout is not None:
            modules.append(nn.Dropout(input_dropout))

        if dims is None:
            dims = [output_size*2]

        dims.insert(0, input_size)

        for (dim1, dim2) in zip(dims[:-1], dims[1:]):
            modules.append(nn.Linear(dim1, dim2))
            modules.append(act_fnc)

        modules.append(nn.Linear(dims[-1], output_size))

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class CustomMLP(nn.Module):
    def __init__(self, input_size, output_size, *args):
        super().__init__()

        self.net = nn.Sequential(
            #nn.Linear(input_size, 1200),
            #nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(1200, output_size * 2),
            nn.ReLU(),
            #nn.Dropout(0.3),
            #nn.Linear(750, 300),
            #nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(output_size * 2, output_size),
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

