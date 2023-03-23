import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from neuron import load_data, normalize_data, flatten_data
import tensorflow as tf
import h5py

from tensorflow import keras
from tensorflow.keras import layers
#from keras import datasets, layers, models
import matplotlib.pyplot as plt

def test_print_dataset(X, y):
    plt.figure(figsize=(10,10))
    for i in range(20):
        plt.subplot(5,5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i])

        plt.xlabel(class_names[y[i][0]])
    plt.show()

if __name__ == "__main__":
    X_train, y_train = load_data('datasets/trainset.hdf5')
    X_test, y_test = load_data('datasets/testset.hdf5')

    print("Shape of X_train before normalize:", X_train.shape)
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    print("Shape of X_train after normalize:", X_train.shape)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    img_size = X_train.shape[1]

    print("Shape of X_train:", X_train.shape)

    X_train = np.expand_dims(X_train, axis=-1)
    X_train = np.tile(X_train, (1, 1, 1, 3))

    X_test = np.expand_dims(X_test, axis=-1)
    X_test = np.tile(X_test, (1, 1, 1, 3))

    print("img_size: ", img_size)
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    class_names = ['cat', 'dog']


    print("class_names: ", len(class_names))

    #test_print_dataset(X_train, y_train)

    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])

    model.summary()

    model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_test, y_test))

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    plt.show()

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

    print(test_acc)

