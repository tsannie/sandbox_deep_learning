import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from neuron import load_data, normalize_data, flatten_data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == "__main__":
    X_train, y_train = load_data('datasets/trainset.hdf5')
    X_test, y_test = load_data('datasets/testset.hdf5')

    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)

    X_test = flatten_data(X_test)
    X_train = flatten_data(X_train)

    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)

    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", name="layer1"),
            layers.Dense(64, activation="relu", name="layer2"),
            layers.Dense(32, activation="relu", name="layer3"),
            layers.Dense(16, activation="relu", name="layer4"),
            layers.Dense(8, activation="relu", name="layer5"),
            layers.Dense(4, activation="relu", name="layer6"),
            layers.Dense(2, activation="relu", name="layer7"),
            layers.Dense(1, activation="sigmoid", name="layer8"),
        ]
    )



    """ pred = predict(X_train, W, b)
    print("Accuracy for train set: ", np.mean(pred == y_train))

    pred = predict(X_test, W, b)
    print("Accuracy for test set: ", np.mean(pred == y_test)) """
