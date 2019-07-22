Simple easy to use module to get the intermediate results from chosen submodules. Supports submodule annidation.

#### Example

```python
model = MyModel()
return_layers = {
    'submodule1': 0,
    'submodule2.conv': 1,
}
mid_getter = IntermediateLayerGetter(module, return_layers, keep_output=False)
mid_outputs, model_output = mid_getter(x)

# mid_outputs = OrderedDict([(0,tensor(...)), (1, tensor(...))])
# model_output is None if keep_ouput is False
# if keep_output is True the model_output contains the final model's output
```