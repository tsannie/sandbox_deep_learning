import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs

def init(dimension):
    params = {}

    for i in range(1, len(dimension)):
        params['W' + str(i)] = np.random.randn(dimension[i], dimension[i - 1])
        params['b' + str(i)] = np.random.randn(dimension[i], 1)

    return params

def forward_propagation(X, params):
    C = len(params) // 2
    activation = {'A0': X}

    for i in range(1, C + 1):
        Z = np.dot(params['W' + str(i)], activation['A' + str(i - 1)]) + params['b' + str(i)]
        A = 1 / (1 + np.exp(-Z))
        activation['A' + str(i)] = A

    return activation

def back_propagation(y, params, activation):
    m = y.shape[1]
    C = len(params) // 2
    grads = {}

    dZ = activation['A' + str(C)] - y
    for i in reversed(range(1, C + 1)):
        dW = 1 / m * np.dot(dZ, activation['A' + str(i - 1)].T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if i > 1:
          dZ = np.dot(params['W' + str(i)].T, dZ) * activation['A' + str(i - 1)] * (1 - activation['A' + str(i - 1)])

        grads['dW' + str(i)] = dW
        grads['db' + str(i)] = db

    return grads

def update(params, grads, alpha):
    C = len(params) // 2

    for i in range(1, C + 1):
        params['W' + str(i)] = params['W' + str(i)] - alpha * grads['dW' + str(i)]
        params['b' + str(i)] = params['b' + str(i)] - alpha * grads['db' + str(i)]

    return params

def predict(X, params):
    activation = forward_propagation(X, params)
    A = activation['A' + str(len(params) // 2)]
    y_pred = np.round(A)
    return y_pred

def loss(A2, y, epsilon=1e-12):
    m = y.shape[1]
    A2 = np.clip(A2, epsilon, 1 - epsilon)
    J = -1 / m * np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2))
    return J

def fit(X, y, alpha=0.1, epochs=100, layers=(32, 32, 32)):
    dimension = list(layers)
    dimension.insert(0, X.shape[0])
    dimension.append(y.shape[0])
    params = init(dimension)

    hist_loss = []
    hist_acc = []

    for i in tqdm(range(epochs)):
        activation = forward_propagation(X, params)
        grads = back_propagation(y, params, activation)
        params = update(params, grads, alpha)

        if not i % 10:
            C = len(params) // 2
            J = loss(activation['A' + str(C)], y)
            hist_loss.append(J)
            pred = predict(X, params)
            hist_acc.append(np.mean(pred == y))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(hist_loss)
    plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(hist_acc)
    plt.title("Accuracy")
    plt.show()

    return params

if __name__ == "__main__":
    X, y = make_blobs(n_samples=1000, centers=2, cluster_std=2, random_state=42)
    print(y)

    X = X.T
    y = y.reshape(1, -1)

    params = fit(X, y, alpha=0.1, epochs=5000, layers=(32, 32, 32))

    pred = predict(X, params)
    print("Accuracy for train set: ", np.mean(pred == y))

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(X[0, :], X[1, :], c=y, cmap="Spectral")
    plt.title("Original data")

    plt.subplot(1, 2, 2)
    x_min, x_max = X[0, :].min() - 0.1, X[0, :].max() + 0.1
    y_min, y_max = X[1, :].min() - 0.1, X[1, :].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 1000), np.linspace(y_min, y_max, 1000))
    Z = predict(np.c_[xx.ravel(), yy.ravel()].T, params)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap="Spectral")
    plt.contour(xx, yy, Z, linewidths=2, colors="black")

    plt.scatter(X[0, :], X[1, :], c=y, cmap="Spectral")
    plt.title("Predicted data")
    plt.show()

    print(X.shape)
    print(y.shape)

