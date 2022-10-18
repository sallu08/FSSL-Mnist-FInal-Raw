import os
import tensorflow as tf
import flwr as fl
from tensorflow import keras
import tensorflow.keras.layers
import numpy as np
import keras.backend as K
import pickle as pkl
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



def softmax(x, temperature=2):
     return np.exp(x/temperature)/sum(np.exp(x/temperature))
 
def myce(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

with open("traina2.pkl", "rb") as f:
    x_trainm, y_trainm = pkl.load(f)

Ma=np.zeros((10,10))
num_classes = 10
input_shape = (28, 28, 1)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Make TensorFlow logs less verbose


def load_partition():
    """Load training and test data to simulate a partition."""
    
    with open("traina2.pkl", "rb") as f:
        x_train, y_train = pkl.load(f)
    with open("testa2.pkl", "rb") as f1:
        x_test, y_test = pkl.load(f1)
    x_train=x_train
    return (
        x_train,
        y_train,
    ), (
        x_test,
        y_test,
    )

        
        
# Define Flower client
class My_Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # current_round=config["current_round"]
        
        # if(current_round<30):
            
        #     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
           
        #     save_weights_only=True,
        #     monitor='val_accuracy',
        #     mode='max',
        #     save_best_only=True)
        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            # callbacks=[model_checkpoint_callback],
            validation_split=0.1,
        )
        hist_df = pd.DataFrame(history.history) 

        hist_csv_file = 'historya.csv'
        with open(hist_csv_file, mode='a') as f:
            hist_df.to_csv(f)
 

        
        (x_trainm, y_trainm), (x_testm, y_testm) = load_partition()        
        
        
        c0_indexes = np.where((y_trainm == 0 ))
        c1_indexes = np.where((y_trainm == 1 ))
        c2_indexes = np.where((y_trainm == 2 ))
        c3_indexes = np.where((y_trainm == 3 ))
        c4_indexes = np.where((y_trainm == 4 ))
        c5_indexes = np.where((y_trainm == 5 ))
        c6_indexes = np.where((y_trainm == 6 ))
        c7_indexes = np.where((y_trainm == 7 ))
        c8_indexes = np.where((y_trainm == 8 ))
        c9_indexes = np.where((y_trainm == 9 ))
      
        xtrain_c0=x_trainm[c0_indexes]
        xtrain_c1=x_trainm[c1_indexes]
        xtrain_c2=x_trainm[c2_indexes]
        xtrain_c3=x_trainm[c3_indexes]
        xtrain_c4=x_trainm[c4_indexes]
        xtrain_c5=x_trainm[c5_indexes]
        xtrain_c6=x_trainm[c6_indexes]
        xtrain_c7=x_trainm[c7_indexes]
        xtrain_c8=x_trainm[c8_indexes]
        xtrain_c9=x_trainm[c9_indexes]
        
        ytrain_c0=y_trainm[c0_indexes]
        ytrain_c1=y_trainm[c1_indexes]
        ytrain_c2=y_trainm[c2_indexes]
        ytrain_c3=y_trainm[c3_indexes]
        ytrain_c4=y_trainm[c4_indexes]
        ytrain_c5=y_trainm[c5_indexes]
        ytrain_c6=y_trainm[c6_indexes]
        ytrain_c7=y_trainm[c7_indexes]
        ytrain_c8=y_trainm[c8_indexes]
        ytrain_c9=y_trainm[c9_indexes]       
        
        N0=ytrain_c0.size
        N1=ytrain_c1.size
        N2=ytrain_c2.size
        N3=ytrain_c3.size
        N4=ytrain_c4.size
        N5=ytrain_c5.size
        N6=ytrain_c6.size
        N7=ytrain_c7.size
        N8=ytrain_c8.size
        N9=ytrain_c9.size
        
        
        layer_name = 'ml'
        intermediate_layer_model = keras.Model(inputs=self.model.input,
                                            outputs=self.model.get_layer(layer_name).output)
        
        feature_c0 = intermediate_layer_model(xtrain_c0).numpy()
        feature_c1 = intermediate_layer_model(xtrain_c1).numpy()
        feature_c2 = intermediate_layer_model(xtrain_c2).numpy()
        feature_c3 = intermediate_layer_model(xtrain_c3).numpy()
        feature_c4 = intermediate_layer_model(xtrain_c4).numpy()
        feature_c5 = intermediate_layer_model(xtrain_c5).numpy()
        feature_c6 = intermediate_layer_model(xtrain_c6).numpy()
        feature_c7 = intermediate_layer_model(xtrain_c7).numpy()
        feature_c8 = intermediate_layer_model(xtrain_c8).numpy()
        feature_c9 = intermediate_layer_model(xtrain_c9).numpy()

        v0=(feature_c0).sum(axis=0)/N0
        v1=(feature_c1).sum(axis=0)/N1
        v2=(feature_c2).sum(axis=0)/N2
        v3=(feature_c3).sum(axis=0)/N3
        v4=(feature_c4).sum(axis=0)/N4
        v5=(feature_c5).sum(axis=0)/N5
        v6=(feature_c6).sum(axis=0)/N6
        v7=(feature_c7).sum(axis=0)/N7
        v8=(feature_c8).sum(axis=0)/N8
        v9=(feature_c9).sum(axis=0)/N9

        s0=softmax(v0)
        s1=softmax(v1)
        s2=softmax(v2)
        s3=softmax(v3)
        s4=softmax(v4)
        s5=softmax(v5)
        s6=softmax(v6)
        s7=softmax(v7)
        s8=softmax(v8)
        s9=softmax(v9)
        global Ma
        Ma=np.column_stack((s0,s1,s2,s3,s4,s5,s6,s7,s8,s9))
        
        np.savetxt("Ma.csv", Ma, delimiter=",")

        
        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["acc"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_acc"][0],
        }
        return parameters_prime, num_examples_train, results



    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        # steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, batch_size=64)
        
        num_examples_test = len(self.x_test)
        # model.fit(x_test, y_test, batch_size=64, epochs=num_epochs, verbose=1)

        return loss, num_examples_test, {"Global accuracy on clientA": accuracy}


def maina() -> None:

    # Load and compile Keras model

    # model = load_model('my_modeli.h5')
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes,activation='relu',name='ml'),

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
                  loss=myce, metrics=['acc'])

    # Load a subset of data to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition()

    # Start Flower client
    client = My_Client(model, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
    )

if __name__ == "__main__":
    maina()
