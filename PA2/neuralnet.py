################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalize your inputs here and return them.
    """
    img = img /255
    return img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels.T] = 1
    labels = one_hot
    return labels


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    # raise NotImplementedError("Softmax not implemented")
    shifted_values = x - np.max(x)
    exp_values = np.exp(shifted_values)
    return exp_values / np.sum(exp_values)


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement the sigmoid activation here.
        """
        # raise NotImplementedError("Sigmoid not implemented")
        self.x = x 
        return 1 / (1 + np.exp(-x))
    def tanh(self, x):
        """
        Implement tanh here.
        """
        # raise NotImplementedError("Tanh not implemented")
        self.x = x
        return 2 / (1 + np.exp(-x)) - 1

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        # raise NotImplementedError("ReLu not implemented")
        self.x = x
        mask = (x >= 0)
        return x * mask

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        # raise NotImplementedError("Sigmoid gradient not implemented")
        return (1 / (1 + np.exp(-self.x))) * (1 - (1 / (1 + np.exp(-self.x))))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        # raise NotImplementedError("tanh gradient not implemented")
        return 1 - (2 / (1 + np.exp(-self.x)) - 1) ** 2

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        # raise NotImplementedError("ReLU gradient not implemented")
        mask = np.ones(self.x.shape) * (self.x > 0)
        return mask
        



class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)    # Declare the Weight matrix
        self.b = np.ones(out_units)    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        # raise NotImplementedError("Layer forward pass not implemented.")
        print("inputs shape: " x.shape)
        x = np.matmul(x,self.w) + self.b
        self.a = x

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # raise NotImplementedError("Backprop for Layer not implemented.")
        print("delta shape: ")
        self.d_w = self.x.T
        self.d_b = np.sum(delta, axis=0)
        self.d_x = self.w.T

class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        self.config = config

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

        print("neural network layers: ", len(self.layers))
    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        if targets == None:
            return x 
        else:
            return x, self.loss(x, targets)       

        raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''

        lambda_l2 = self.config['L2_penalty']
        loss = 0
        # raise NotImplementedError("Loss not implemented for NeuralNetwork")
        for i in range(len(self.layers)):
            if i % 2 == 0:
                loss += lambda_l2 * np.linalg.norm(self.layers[i].w)

        
        labels_onehot = one_hot_encoding(targets)
        cross_entropy = labels_onehot * np.log(softmax(logits))
        loss += -np.mean(cross_entropy)
        self.loss = loss
        return loss

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        # raise NotImplementedError("Backprop not implemented for NeuralNetwork")
        
        for i in range(len(self.layers)):
            

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    
    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """

    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    num_train_data = x_train.shape[0]
    split = int(num_train_data*0.2)
    rand_id = np.random.permutation(np.arange(num_train_data))
    val_id = rand_id[:split]
    train_id = rand_id[split:]

    x_valid, y_valid = x_train[val_id], y_train[val_id]
    x_train, y_train = x_train[train_id], y_train[train_id]

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
