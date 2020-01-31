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
    one_hot[np.arange(len(labels)),labels] = 1
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
    # print("shifted values: ", shifted_values)
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
        self.z = x 
        return 1 / (1 + np.exp(-x))
    def tanh(self, x):
        """
        Implement tanh here.
        """
        # raise NotImplementedError("Tanh not implemented")
        self.z = x
        return 2 / (1 + np.exp(-2*x)) - 1

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        # raise NotImplementedError("ReLu not implemented")
        self.z = x
        mask = (x >= 0)
        return x * mask

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        # raise NotImplementedError("Sigmoid gradient not implemented")
        return (1 / (1 + np.exp(-self.z))) * (1 - (1 / (1 + np.exp(-self.z))))

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        # raise NotImplementedError("tanh gradient not implemented")
        return 1 - (2 / (1 + np.exp(-2*self.z)) - 1) ** 2

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        # raise NotImplementedError("ReLU gradient not implemented")
        mask = np.ones(self.z.shape) * (self.z > 0)
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
        self.b = np.zeros(out_units)    # Create a placeholder for Bias
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
        self.x = x
        self.a = np.matmul(x,self.w) + self.b

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # raise NotImplementedError("Backprop for Layer not implemented.")
        self.d_w = np.dot(self.x.T, delta)
        # print("dw: ", self.d_w)
        self.d_b = np.sum(delta, axis=0)
        self.d_x = np.dot(delta, self.w.T)

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
            if i % 2 == 0:
                self.layers[i](x)
                x = self.layers[i].a
            else:
                x = self.layers[i](x)
        if type(targets) == None:
            return x 
        else:
            return x, self.loss(x, targets)       

        raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        self.logits = logits
        self.targets = targets

        lambda_l2 = self.config['L2_penalty']
        loss = 0
        # raise NotImplementedError("Loss not implemented for NeuralNetwork")
        for i in range(len(self.layers)):
            if i % 2 == 0:
                loss += lambda_l2 * np.linalg.norm(self.layers[i].w)**2 / 2
        

        cross_entropy = targets * np.log(softmax(logits))
        loss += -np.mean(cross_entropy)
        # print("loss: ", loss)
        self.loss_value = loss
        self.targets = targets

        return loss

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        # raise NotImplementedError("Backprop not implemented for NeuralNetwork")
        lr = self.config['learning_rate']
        lambda_l2 = self.config['L2_penalty']

        N = len(self.targets)
        # shifted_values = self.logits - np.max(self.logits)
        # exp_logits = np.exp(shifted_values)
        exp_logits = np.exp(self.logits)
        sum_exp_logits = np.sum(exp_logits, axis=1)
        delta = ((sum_exp_logits).reshape(-1,1) * self.targets - exp_logits) / sum_exp_logits.reshape(-1,1)
        delta = - delta / N
        # print("delta: ", delta)

        for i in range(len(self.layers)-1, -1, -1):
            if i%2 == 0:
                self.layers[i].backward(delta)
                self.layers[i].w = self.layers[i].w - lr * (self.layers[i].d_w + lambda_l2 * self.layers[i].w)
                self.layers[i].b -= lr * self.layers[i].d_b
                delta = self.layers[i].d_x
            else:
                delta = self.layers[i].backward(delta)


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    EPOCH = config['epochs']
    BATCH_SIZE = config['batch_size']
    num_batch_train = int(x_train.shape[0] / BATCH_SIZE)
    num_batch_val = int(x_valid.shape[0] / BATCH_SIZE)

    train_loss_his = []
    train_acc_his = []
    
    val_loss_his = []
    val_acc_his = []

    for epoch in range(EPOCH):

        #########################################################################
        ##########################       TRAINING         #######################
        #########################################################################
        for iter in range(num_batch_train-1):
            inputs = x_train[iter*BATCH_SIZE:(iter+1)*BATCH_SIZE]
            targets = y_train[iter*BATCH_SIZE:(iter+1)*BATCH_SIZE]

            outputs = model(inputs)
            loss = model.loss(outputs, targets)

            model.backward()

            preds_one_hot = one_hot_encoding(np.argmax(outputs, axis=1))
            acc = np.sum((preds_one_hot*targets))/BATCH_SIZE
            train_loss_his.append(loss)
            train_acc_his.append(acc)
        
        # final batch
        inputs = x_train[iter*BATCH_SIZE:]
        targets = y_train[iter*BATCH_SIZE:]
        final_size = inputs.shape[0]
        
        outputs = model(inputs)
        loss = model.loss(outputs, targets)

        model.backward()
        
        preds_one_hot = one_hot_encoding(np.argmax(outputs, axis=1))
        acc = np.sum((preds_one_hot*targets))/final_size
        train_loss_his.append(loss)
        train_acc_his.append(acc)

        train_loss_mean = np.mean(np.array(train_loss_his))
        train_acc_mean = np.mean(np.array(train_acc_his))

        for iter in range(num_batch_val-1):
            inputs = x_valid[iter*BATCH_SIZE:(iter+1)*BATCH_SIZE]
            targets = y_valid[iter*BATCH_SIZE:(iter+1)*BATCH_SIZE]

            outputs = model(inputs)
            loss = model.loss(outputs, targets)

            preds_one_hot = one_hot_encoding(np.argmax(outputs, axis=1))
            acc = np.sum((preds_one_hot*targets)) / BATCH_SIZE

            val_loss_his.append(loss)
            val_acc_his.append(acc)
        
        # final batch
        inputs = x_valid[iter*BATCH_SIZE:]
        targets = y_valid[iter*BATCH_SIZE:]
        final_size = inputs.shape[0]
        
        outputs = model(inputs)
        loss = model.loss(outputs, targets)
        
        preds_one_hot = one_hot_encoding(np.argmax(outputs, axis=1))
        acc = np.sum((preds_one_hot*targets))/final_size
        val_loss_his.append(loss)
        val_acc_his.append(acc)

        val_loss_mean = np.mean(np.array(val_loss_his))
        val_acc_mean = np.mean(np.array(val_acc_his))

        print("val acc: ", val_acc_mean)
            
    # raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """

    batch_size = 128
    num_batch_test = int(X_test.shape[0] / batch_size)
    
    test_acc_his = []

    for i in range(num_batch_test):
        inputs = X_test[i*batch_size: (i+1)*batch_size]
        targets = y_test[i*batch_size: (i+1)*batch_size]

        outputs = model(inputs)
        
        preds_one_hot = one_hot_encoding(np.argmax(outputs, axis=1))
        acc = np.sum((preds_one_hot * targets)) / batch_size

        test_acc_his.append(acc)
    
    inputs = X_test[i*batch_size:]
    targets = y_test[i*batch_size:]
    
    final_size = inputs.shape[0]

    outputs = model(inputs)

    preds_one_hot = one_hot_encoding(np.argmax(outputs, axis=1))
    acc = np.sum((preds_one_hot * targets)) / batch_size
    test_acc_his.append(acc)

    test_acc_mean = np.mean(np.array(test_acc_his))
    print("test acc: ", test_acc_mean)

    return test_acc_mean
    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")
    print("config: ", config)

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
