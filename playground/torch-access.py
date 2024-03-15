"""
PyTorch Model Access
"""

# Module Imports
from torch.nn import Sequential, Linear, ReLU, Sequential

# Model Parameters
input_size = 10
hidden_size = [48, 32, 16, 8, 4]
output_size = 1
bias = True

# Model Definition
model = Sequential(
    Linear(input_size, hidden_size[0], bias=bias),
    ReLU(),
    Linear(hidden_size[0], hidden_size[1], bias=bias),
    ReLU(),
    Linear(hidden_size[1], hidden_size[2], bias=bias),
    ReLU(),
    Linear(hidden_size[2], hidden_size[3], bias=bias),
    ReLU(),
    Linear(hidden_size[3], hidden_size[4], bias=bias),
    ReLU(),
    Linear(hidden_size[4], output_size, bias=bias)
)

if __name__ == "__main__":
    input_size_from_model = model[0].in_features
    hidden_size_from_model = []
    output_size_from_model = model[-1].out_features

    for name, param in model.named_parameters():
        if 'weight' in name:
            hidden_size_from_model.append(param.shape[0])

    print(f"Input Size: {input_size_from_model}")
    print(f"Hidden Size: {hidden_size_from_model[0:-1]}")
    print(f"Output Size: {output_size_from_model}")