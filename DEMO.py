import torch
import torch.nn as nn

from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.nested = nn.Sequential(
            nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 3)),
            nn.Linear(3, 1),
        )
        self.interaction_idty = nn.Identity()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)

        interaction = x1 * x2
        self.interaction_idty(interaction)

        x_out = self.nested(interaction)

        return x_out

if __name__ == '__main__':
    model = Model()
    return_layers = {
        'fc2': 'fc2',
        'nested.0.1': 'nested',
        'interaction_idty': 'interaction',
    }
    mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
    mid_outputs, model_output = mid_getter(torch.randn(1, 2))

    print(model_output)
    print(mid_outputs)

