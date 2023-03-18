import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import h5py
from tqdm import tqdm

def init(X):
    n = X.shape[1]
    W = np.random.randn(n, 1)
    b = np.random.randn(1)
    return W, b

def model(X, W, b):
    Z = np.dot(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def loss(A, y, epsilon=1e-12):
    m = y.shape[0]
    A = np.clip(A, epsilon, 1 - epsilon)
    J = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    return J

def gradient(X, y, A):
    m = y.shape[0]
    dW = 1 / m * np.dot(X.T, A - y)
    db = 1 / m * np.sum(A - y)
    return dW, db

def update(W, b, dW, db, alpha):
    W = W - alpha * dW
    b = b - alpha * db
    return W, b

def predict(X, W, b):
    A = model(X, W, b)
    y_pred = np.round(A)
    return y_pred

def fit(X, y, alpha=0.1, epochs=100):
    W, b = init(X)

    hist_loss = []
    hist_acc = []

    for i in tqdm(range(epochs)):
        A = model(X, W, b)

        if not i % 100:
            J = loss(A, y)
            hist_loss.append(J)
            pred = predict(X, W, b)
            hist_acc.append(np.mean(pred == y))

        dW, db = gradient(X, y, A)
        W, b = update(W, b, dW, db, alpha)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(hist_loss)
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(hist_acc)
    plt.title("Accuracy")
    plt.show()
    return W, b

def load_data(path):
    try:
        train_dataset = h5py.File(path, "r")
        header = list(train_dataset.keys())
        X = np.array(train_dataset[header[0]][:])
        y = np.array(train_dataset[header[1]][:])
    except Exception as e:
        print("Error loading data: ", e)

    return X, y

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def flatten_data(data):
    return data.reshape(data.shape[0], -1)

if __name__ == "__main__":
    X_train, y_train = load_data('datasets/trainset.hdf5')
    X_test, y_test = load_data('datasets/testset.hdf5')
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)

    print("X_max: ", np.max(X_train))
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    print("X_max: ", np.max(X_train))


    X_test = flatten_data(X_test)
    X_train = flatten_data(X_train)

    print("X_train: ", X_train.shape)

    W, b = fit(X_train, y_train, alpha=0.01, epochs=10000)
    print("W: ", W.shape)
    print("b: ", b.shape)
    pred = predict(X_train, W, b)
    print("Accuracy for train set: ", np.mean(pred == y_train))

    pred = predict(X_test, W, b)
    print("Accuracy for test set: ", np.mean(pred == y_test))


    """     plt.figure(figsize=(16, 8))
    for i in range(10):
        plt.subplot(4, 5, i + 1)
        plt.imshow(X_train[i], cmap='gray')
        plt.tight_layout()
    plt.show() """

    """ W, b = fit(X_train, y_train, alpha=0.1, epochs=100)

    pred = predict(X_test, W, b) """
    #print("Accuracy: ", np.mean(pred == y_test))
