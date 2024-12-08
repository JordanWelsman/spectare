# TensorFlow/Keras Test Models

def main():
    # Import TensorFlow/Keras submodules if installed
    try:
        import tensorflow as tf
        KERAS_EXISTS = True
    except ImportError:
        KERAS_EXISTS = False


    # Build models if TensorFlow/Keras is installed
    if KERAS_EXISTS:
        print("Loading TensorFlow/Keras models...")
        model_1 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(3, activation='relu'),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

        model_2 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

        model_3 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(4, activation='relu'),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(2, activation='relu'),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

        model_4 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(6, activation='relu'),
            tf.keras.layers.Dense(7, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(7, activation='relu'),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

        model_5 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

        # Compile Models
        for model in [model_1, model_2, model_3, model_4, model_5]:
            for layer in model.layers:
                if hasattr(layer, "bias"):
                    bias_shape = layer.bias.shape
                    # Generate random biases using a normal distribution
                    random_biases = tf.random.normal(shape=bias_shape)
                    layer.bias.assign(random_biases)
            model.compile(optimizer='adam', loss='mean_squared_error')

        __all__ = ["model_1", "model_2", "model_3", "model_4", "model_5"]
        print("TensorFlow/Keras models loaded.")

        filename = "model.h5"
        model_1.save(filename)
        print(f"Model 1 saved as {filename}.")

    else:
        print("TensorFlow/Keras not installed. Skipping model loading.")


# Runtime
if __name__ == "__main__":
    main()