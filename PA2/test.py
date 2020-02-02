from neuralnet import *
import matplotlib.pyplot as plt


def train2(model, x_train, y_train, x_valid, y_valid, labels_train, labels_valid, config):
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

    train_losses, train_acces, valid_losses, valid_acces = [], [], [], []

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

        '''
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
        acc = np.mean((preds==labels))

        train_loss_his.append(loss)
        train_acc_his.append(acc)
        '''

        train_loss_mean = np.mean(np.array(train_loss_his))
        train_acc_mean = np.mean(np.array(train_acc_his))
        print("train loss: ", train_loss_mean)

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

        train_losses.append(train_loss_mean)
        train_acces.append(train_acc_mean)
        valid_losses.append(val_loss_mean)
        valid_acces.append(val_acc_mean)

        # print("val loss: ", val_loss_mean)
    return train_losses, train_acces, valid_losses, valid_acces

    # raise NotImplementedError("Train method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model = Neuralnetwork(config)

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

    # train the model
    train_losses, train_acces, valid_losses, valid_acces = train2(
        model, x_train, y_train, x_valid, y_valid, labels_train, labels_valid, config)

    plt.plot(list(range(len(train_losses))),
             train_losses, label='Training Loss')
    plt.plot(list(range(len(train_losses))),
             valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(list(range(len(train_losses))),
             train_acces, label='Training Accuracy')
    plt.plot(list(range(len(train_losses))), valid_acces,
             label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    test_acc = test(model, x_test, y_test)
