Simple easy to use module to get the intermediate results from chosen submodules. Supports submodule annidation. Inspired in [this](https://github.com/pytorch/vision/blob/f76e598d47879dbd917bf5936bbd11ff41632787/torchvision/models/_utils.py#L7) but does not assume that submodules are executed sequentially.

#### Example

```python
model = MyModel()
# Model definition
# MyModel(
#   (submodule1): Submodule(...)
#   (submodule2): Submodule(
#       conv: Conv2(...)
#   )
# ...
# )

# return_layers: {[current_module_name]: [desired_output_name]}
return_layers = {
    'submodule1': 0,
    'submodule2.conv': 'conv',
}
mid_getter = IntermediateLayerGetter(module, return_layers, keep_output=False)
mid_outputs, model_output = mid_getter(x)

# mid_outputs = OrderedDict([(0,tensor(...)), ('conv', tensor(...))])
# model_output is None if keep_ouput is False
# if keep_output is True the model_output contains the final model's output
```
