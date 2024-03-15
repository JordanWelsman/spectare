"""
Testing TensorFlow/Keras Model States
"""

# Module Imports
import tensorflow as tf

# Model Hyperparameters
INPUT_SIZE = 3
HIDDEN_SIZE = 2
OUTPUT_SIZE = 1

# Sequential Model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(HIDDEN_SIZE, input_shape=(INPUT_SIZE,)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(OUTPUT_SIZE)
])