# PyTorch Example Models

# Import PyTorch submodules if installed
try:
    from torch.nn import Sequential, Linear, ReLU
    TORCH_EXISTS = True
except ImportError:
    TORCH_EXISTS = False


# Build models if PyTorch is installed

if TORCH_EXISTS:
    model_1 = Sequential(
        Linear(2, 3, bias=True),
        ReLU(),
        Linear(3, 1, bias=True),
    )

    model_2 = Sequential(
        Linear(3, 4, bias=True),
        ReLU(),
        Linear(4, 4, bias=True),
        ReLU(),
        Linear(4, 1, bias=True),
    )

    model_3 = Sequential(
        Linear(3, 4, bias=True),
        ReLU(),
        Linear(4, 5, bias=True),
        ReLU(),
        Linear(5, 2, bias=True),
        ReLU(),
        Linear(2, 1, bias=True),
    )

    model_4 = Sequential(
        Linear(5, 6, bias=True),
        ReLU(),
        Linear(6, 7, bias=True),
        ReLU(),
        Linear(7, 8, bias=True),
        ReLU(),
        Linear(8, 8, bias=True),
        ReLU(),
        Linear(8, 7, bias=True),
        ReLU(),
        Linear(7, 1, bias=True)
    )

    model_5 = Sequential(
        Linear(10, 12, bias=True),
        ReLU(),
        Linear(12, 12, bias=True),
        ReLU(),
        Linear(12, 12, bias=True),
        ReLU(),
        Linear(12, 12, bias=True),
        ReLU(),
        Linear(12, 12, bias=True),
        ReLU(),
        Linear(12, 1, bias=True)
    )

    __all__ = ["model_1", "model_2", "model_3", "model_4", "model_5"]

    print("PyTorch models loaded.")

else:
    print("PyTorch not installed. Skipping model loading.")