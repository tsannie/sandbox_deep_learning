# Description

Welcome to my personal git repository for deep learning training. Here, I am experimenting with different neural network architectures and deep learning algorithms to understand how they work.

## Program 1: Neuron Recognition Algorithm (neuron.py)

### Objective

The objective of this program is to implement a recognition algorithm for distinguishing between dogs and cats using a single neuron.

### Conclusion

After training the algorithm, we observed that the accuracy of the train set was not bad. However, we also encountered the problem of overfitting. To overcome this, we need to increase the number of examples in our dataset significantly. Additionally, since a single neuron is linearly separable, we cannot solve this problem using a single neuron. Hence, we need to start building a network.

Let's move on to building a neural network to solve this problem!

## Program 2: Neural Network (neural_network.py)

### Objective

The objective of this program is to implement a neural network. We will use sklearn to build datasets and train the neural network.

### Exemplary Output

![image](https://i.imgur.com/MdLglIA.png)
![image](https://i.imgur.com/QKBqdfr.png)
![image](https://i.imgur.com/9rCYBMW.png)

### Conclusion

In conclusion, the neural network implementation works well as shown in the exemplary output, and it proves that the problem is no longer linearly separable. However, it still doesn't work for the cat-dog dataset due to overfitting. Moreover, we observed that the performance of the model does not increase with the number of layers inside, which could be due to the vanishing gradient problem where the gradients become too small during backpropagation, hindering the network's ability to learn.

## Program 3: Convolutional Neural Network (cnn.py)

### Objective

The objective of this program is to implement a convolutional neural network. We will use keras to build datasets and train the neural network.
