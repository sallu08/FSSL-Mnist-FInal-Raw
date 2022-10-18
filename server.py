from typing import Dict, Optional, Tuple
import flwr as fl
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evalation
    input_shape = (28, 28, 1)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), padding='same',
                               activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu', name='ml'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy',
                  metrics=['acc'], weighted_metrics=[])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.25,
        fraction_evaluate=0.25,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        # evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(
            model.get_weights()),

    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy)


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

    x_val, y_val = x_train[48000:50000], y_train[48000:50000]
    x_val = x_val.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
    x_val = np.expand_dims(x_val, -1)
    y_val = keras.utils.to_categorical(y_val, 10)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        model.set_weights(parameters)
        # evaluate model at server level
        loss, accuracy = model.evaluate(x_val, y_val)
        # model.save('my_modeli.h5')
        # model.save_weights('my_modeli_weights.h5')

        return loss, {"accuracy on server": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 64,
        "current_round": server_round,
        "local_epochs": 1  # if server_round < 2 else 3,

    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5  #   if server_round < 5 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
