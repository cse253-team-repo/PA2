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

import os
import gzip
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
    img = img / 255
    return img


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    """
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
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
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels, labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.
    """
    # raise NotImplementedError("Softmax not implemented")
    shifted_values = x - np.max(x, axis=1).reshape(-1, 1)
    # print("shifted values: ", shifted_values)
    exp_values = np.exp(shifted_values)
    return exp_values / np.sum(exp_values, axis=1).reshape(-1, 1)


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type="sigmoid"):
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
        self.z = x
        return 1 / (1 + np.exp(-x))
        raise NotImplementedError("Sigmoid not implemented")

    def tanh(self, x):
        """
        Implement tanh here.
        """
        self.z = x
        return 2 / (1 + np.exp(-2*x)) - 1
        raise NotImplementedError("Tanh not implemented")

    def ReLU(self, x):
        """
        Implement ReLU here.
        """
        x = x / (np.max(x)-np.min(x))
        self.z = x
        mask = (x >= 0)
        return x * mask
        raise NotImplementedError("ReLu not implemented")

    def grad_sigmoid(self):
        """
        Compute the gradient for sigmoid here.
        """
        return (1 / (1 + np.exp(-self.z))) * (1 - (1 / (1 + np.exp(-self.z))))
        raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        return 1 - (2 / (1 + np.exp(-2*self.z)) - 1) ** 2
        raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        mask = np.ones(self.z.shape) * (self.z > 0)
        return mask
        raise NotImplementedError("Sigmoid gradient not implemented")


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
        # Declare the Weight matrix
        self.w = np.random.randn(in_units, out_units)
        self.b = np.zeros(out_units)    # Create a placeholder for Bias
        self.x = None    # Save the input to forward in this
        # Save the output of forward pass in this (without activation)
        self.a = None

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
        self.a = np.dot(x, self.w) + self.b

    def backward(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # raise NotImplementedError("Backprop for Layer not implemented.")
        self.d_w = np.dot(self.x.T, delta)
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
        self.momentum = self.config['momentum']
        if self.momentum:
            self.gamma = self.config['momentum_gamma']
            self.v_w = [0] * (len(config['layer_specs'])+1)
            self.v_b = [0] * (len(config['layer_specs'])+1)
        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(
                Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
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
        if type(targets) != type(None):
            return x, self.loss(x, targets)
        else:

            return x

        raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        self.logits = logits
        self.targets = targets

        lambda_l2 = self.config['L2_penalty']
        loss_value = 0
        # raise NotImplementedError("Loss not implemented for NeuralNetwork")
        for i in range(len(self.layers)):
            if i % 2 == 0:
                loss_value += lambda_l2 * \
                    np.linalg.norm(
                        self.layers[i].w)**2 / (2 * (self.layers[i].w.shape[0] * self.layers[i].w.shape[1]))
        cross_entropy = - np.mean(targets * np.log(softmax(logits)))

        loss_value += cross_entropy
        self.targets = targets

        return loss_value

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        # raise NotImplementedError("Backprop not implemented for NeuralNetwork")

        N = len(self.targets)
        # shifted_values = self.logits - np.max(self.logits)
        # exp_logits = np.exp(shifted_values)
        exp_logits = np.exp(self.logits)
        sum_exp_logits = np.sum(exp_logits, axis=1)
        # # print("sum_Exp_logits: ", sum_exp_logits)
        delta = ((sum_exp_logits).reshape(-1, 1) * self.targets -
                 exp_logits) / sum_exp_logits.reshape(-1, 1)
        # print("delta: ", delta)
        # print("targets shape: ", self.targets.shape)
        # print("porb shape: ", softmax(self.logits).shape)
        # delta = self.targets - softmax(self.logits)
        # delta = -delta
        # self.delta = delta
        for i in range(len(self.layers)-1, -1, -1):
            if i % 2 == 0:
                self.layers[i].backward(delta)
                delta = self.layers[i].d_x
            else:
                delta = self.layers[i].backward(delta)

    def update(self):
        lr = self.config['learning_rate']
        lambda_l2 = self.config['L2_penalty']
        if self.momentum:
            for i in range(len(self.layers)-1, -1, -1):
                if i % 2 == 0:
                    self.v_w[i] = self.gamma * self.v_w[i] + \
                        (1-self.gamma) * \
                        (self.layers[i].d_w - lambda_l2 * self.layers[i].w)
                    self.layers[i].w += lr * self.v_w[i]

                    self.v_b[i] = self.gamma * self.v_b[i] + \
                        (1-self.gamma) * (self.layers[i].d_b)
                    self.layers[i].b += lr * self.v_b[i]
                else:
                    pass
        else:
            for i in range(len(self.layers)-1, -1, -1):
                if i % 2 == 0:
                    self.layers[i].w += lr * \
                        (self.layers[i].d_w - lambda_l2 * self.layers[i].w)
                    self.layers[i].b += lr * self.layers[i].d_b
                else:
                    pass


def train(model, x_train, y_train, x_valid, y_valid, labels_train, labels_valid, config):
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

    for epoch in range(EPOCH):
        train_loss_his = []
        train_acc_his = []

        val_loss_his = []
        val_acc_his = []
        #########################################################################
        ##########################       TRAINING         #######################
        #########################################################################
        for iter in range(num_batch_train-1):
            loss = 0
            inputs = x_train[iter*BATCH_SIZE:(iter+1)*BATCH_SIZE]
            targets = y_train[iter*BATCH_SIZE:(iter+1)*BATCH_SIZE]
            labels = labels_train[iter*BATCH_SIZE:(iter+1)*BATCH_SIZE]

            outputs = model(inputs)
            prob = softmax(outputs)
            preds = np.argmax(prob, axis=1)

            loss = model.loss(outputs, targets)
            model.backward()
            model.update()

            # preds_one_hot = one_hot_encoding(np.argmax(outputs, axis=1))
            # acc = np.sum((preds_one_hot*targets))/BATCH_SIZE
            acc = np.mean((preds == labels))
            train_loss_his.append(loss)
            train_acc_his.append(acc)

        # final batch
        loss = 0
        inputs = x_train[iter*BATCH_SIZE:]
        targets = y_train[iter*BATCH_SIZE:]
        labels = labels_train[iter*BATCH_SIZE:]
        final_size = inputs.shape[0]

        outputs = model(inputs)
        preds = np.argmax(outputs, axis=1)

        loss = model.loss(outputs, targets)

        model.backward()
        model.update()

        # preds_one_hot = one_hot_encoding(np.argmax(outputs, axis=1))
        # acc = np.sum((preds_one_hot*targets))/final_size
        acc = np.mean((preds == labels))

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


def grad_check(model, model_1, model_2,  x_train, labels_train, config):
    for j in range(10):
        sample = x_train[np.where(labels_train == j)[0][5]].reshape(1, -1)
        targets = np.array([j])

        outputs = model(sample)
        loss = model.loss(outputs, targets)
        x = 3
        model.backward()

        # numeric gradient
        for i in range(len(model.layers)-1, -1, -1):
            # for i in range(2):
            if i % 2 == 0:
                model_1.layers[i].w[x][x] = model_1.layers[i].w[x][x] + 1e-2
                model_2.layers[i].w[x][x] = model_2.layers[i].w[x][x] - 1e-2
                outputs_1, outputs_2 = model_1(sample), model_2(sample)
                loss_1, loss_2 = model_1.loss(
                    outputs_1, targets), model_2.loss(outputs_2, targets)
                delta_loss = loss_1 - loss_2
                numeric_grad = delta_loss / 2e-2
                # print("delta loss: ", delta_loss)

                auto_grad = model.layers[i].d_w[x][x]
                print("grad difference: ", auto_grad - numeric_grad)

                model_1.layers[i].w[x][x] = model_1.layers[i].w[x][x] - 1e-2
                model_2.layers[i].w[x][x] = model_2.layers[i].w[x][x] + 1e-2

            else:
                pass

        model.backward()


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model = Neuralnetwork(config)
    model_1 = Neuralnetwork(config)
    model_2 = Neuralnetwork(config)

    # Load the data
    x_train, y_train, labels_train = load_data(path="./", mode="train")
    x_test,  y_test, labels_test = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    num_train_data = x_train.shape[0]
    split = int(num_train_data*0.2)
    rand_id = np.random.permutation(np.arange(num_train_data))
    val_id = rand_id[:split]
    train_id = rand_id[split:]

    x_valid, y_valid, labels_valid = x_train[val_id], y_train[val_id], labels_train[val_id]
    x_train, y_train, labels_train = x_train[train_id], y_train[train_id], labels_train[train_id]

    # gradient check
    grad_check(model, model_1, model_2, x_train, labels_train, config)
    # train the model
    # train(model, x_train, y_train, x_valid, y_valid, labels_train, labels_valid, config)

    # test_acc = test(model, x_test, y_test)
