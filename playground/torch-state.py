"""
Testing PyTorch Model States
"""

# Module Imports
from torch.nn import Linear, Module,  Sequential, ReLU

# Model Hyperparameters
INPUT_SIZE = 3
HIDDEN_SIZE = 2
OUTPUT_SIZE = 1

# Sequential Model
model_1 = Sequential(
    Linear(INPUT_SIZE, HIDDEN_SIZE),
    ReLU(),
    Linear(HIDDEN_SIZE, OUTPUT_SIZE)
)

# Module Model
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.layer2 = Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.relu = ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model_2 = Model()

# Runtime
if __name__ == "__main__":
    print(f"Sequential Model:\n {model_1.state_dict()}")
    print(f"Module Model:\n {model_2.state_dict()}")
