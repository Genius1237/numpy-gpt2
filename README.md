# numpy-gpt2
Trying to implement GPT2 in Numpy from scratch without referring to Huggingface Code, only consulting them for weights (this is probably too lax even, but who cares)

`model.py` contains implementation

`test_init_weights.py` contains tests for weight loading. The weights are loaded from the huggingfce implementation, but follow a dot separated format. That gets parsed and sent into our model

`test_layers.py` tests for invidivual layers implemented in `model.py`

To run tests
```
pytest -s test_layers.py -k gelu
```
`-k` lets you specify a subset of tests to run by matching a substring that is in the test name.