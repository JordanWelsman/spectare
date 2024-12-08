from torch.nn import Sequential, Linear, ReLU
import tensorflow as tf

# PyTorch
t_model = Sequential(
    Linear(2, 3, bias=True),
    ReLU(),
    Linear(3, 1, bias=True)
)

print(f"PyTorch Weights ({type(t_model.state_dict()['0.weight'])}):\n {t_model.state_dict()['0.weight'].tolist()}")

# TensorFlow
k_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)), # Input layer
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax')
])

print(f"TensorFlow Weights ({type(k_model.get_weights()[0])}):")
print(f" Before:\n  {k_model.get_weights()[0].tolist()}")

# transpose weight matrix
k_T_weights = k_model.get_weights()[0].T.tolist()
print(f" After:\n  {k_T_weights}")
