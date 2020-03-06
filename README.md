Simple easy to use Pytorch module to get the intermediate layers outputs from chosen submodules. Inspired in [this](https://github.com/pytorch/vision/blob/f76e598d47879dbd917bf5936bbd11ff41632787/torchvision/models/_utils.py#L7) but does not assume that submodules are executed sequentially. Pypi [link](https://pypi.org/project/torch-intermediate-layer-getter/)

List of features:
- Supports submodule annidation (module1.submodule2.submodule3)
- In case that a module is called more than once during a forward pass, all it's outputs are saved a in a list.

# Installation

```sh
pip install torch-intermediate-layer-getter
```

# Usage
## Example

```python
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
        self.interaction_idty = nn.Identity() # Simple trick for operations not performed as modules

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)

        interaction = x1 * x2
        self.interaction_idty(interaction)

        x_out = self.nested(interaction)

        return x_out
        
model = Model()
return_layers = {
    'fc2': 'fc2',
    'nested.0.1': 'nested',
    'interaction_idty': 'interaction',
}
mid_getter = MidGetter(model, return_layers=return_layers, keep_output=True)
mid_outputs, model_output = mid_getter(torch.randn(1, 2))

print(model_output)
>> tensor([[0.3219]], grad_fn=<AddmmBackward>)
print(mid_outputs)
>> OrderedDict([('fc2', tensor([[-1.5125,  0.9334]], grad_fn=<AddmmBackward>)),
  ('interaction', tensor([[-0.0687, -0.1462]], grad_fn=<MulBackward0>)),
  ('nested', tensor([[-0.1697,  0.1432,  0.2959]], grad_fn=<AddmmBackward>))])

# model_output is None if keep_ouput is False
# if keep_output is True the model_output contains the final model's output
```
