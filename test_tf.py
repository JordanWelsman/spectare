# Module Imports
from time import time
import spectare as sp
import tensorflow as tf

# Runtime
def main(t: bool = False):
    t1 = time()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)), # Input layer
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
    ])
    for layer in model.layers:
        if hasattr(layer, "bias"):
            bias_shape = layer.bias.shape
            # Generate random biases using a normal distribution
            random_biases = tf.random.normal(shape=bias_shape)
            layer.bias.assign(random_biases)
    model.compile(optimizer='adam', loss='mean_squared_error')
    t2 = time()
    print(f"Model Creation Time: {(t2 - t1):.4f}s") if t else None

    model_info = sp.get_tf_model_info(model)
    sp.draw_tf_network(
        num_layers=model_info["num_layers"],
        num_nodes=model_info["num_nodes"],
        model=model,
        filename="TensorFlow Graph.png",
        node_base_size=2000,
        node_size_scaling_factor=640,
        colorblind=True,
        draw_labels=False,
        draw_legend=True,
        white_neutral=True
    )

if __name__ == "__main__":
    main(t=False)