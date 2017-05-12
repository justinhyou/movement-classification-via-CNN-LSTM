# Note that the dataset must be already downloaded for this script to work, do:
#     $ cd data/
#     $ python download_dataset.py

import tensorflow as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn
import os

# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'rb')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))

# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'rb')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # Trainging
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 1
        #self.training_epochs = 240000 --TODO
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: three 3D sensors features over time
        self.n_hidden = 32  # nb of neurons inside the neural network
        self.n_classes = 6  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(feature_mat, config):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # Exchange dim 1 and dim 0
    feature_mat = tf.transpose(feature_mat, [1, 0, 2])
    # New feature_mat's shape: [time_steps, batch_size, n_inputs]

    # Temporarily crush the feature_mat's dimensions
    feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
    # New feature_mat's shape: [time_steps*batch_size, n_inputs]

    # Linear activation, reshaping inputs to the LSTM's number of hidden:
    hidden = tf.nn.relu(tf.matmul(
        feature_mat, config.W['hidden']
    ) + config.biases['hidden'])
    # New feature_mat (hidden) shape: [time_steps*batch_size, n_hidden]

    # Split the series because the rnn cell needs time_steps features, each of shape:
    hidden = tf.split(0, config.n_steps, hidden)
    #hidden = tf.split(hidden, config.n_steps, 0)
    # New hidden's shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_hidden]

    # Define LSTM cell of first hidden layer:
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    #lstm_cell = tf.nn.rnn_cell.BasicLSTMCell.__call__(config.n_hidden, forget_bias=1.0)

    # Stack two LSTM layers, both layers has the same shape
    lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)

    # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
    outputs, _ = tf.nn.rnn(lsmt_layers, hidden, dtype=tf.float32)
    # outputs' shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_classes]

    # Get last time step's output feature for a "many to one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']


def one_hot(label):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    label_num = len(label)
    new_label = label.reshape(label_num)  # shape : [sample_num]
    # because max is 5, and we will create 6 columns
    n_values = np.max(new_label) + 1
    return np.eye(n_values)[np.array(new_label, dtype=np.int32)]


if __name__ == "__main__":

    #-----------------------------
    # step1: load and prepare data
    #-----------------------------
    # Those are separate normalised input features for the neural network
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_"
    ]

    # Output classes to learn how to classify
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    DATA_PATH = "data/"
    DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
    print("\n" + "Dataset is now located at: " + DATASET_PATH)
    TRAIN = "train/"
    TEST = "test/"

    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    y_train = one_hot(load_y(y_train_path))
    y_test = one_hot(load_y(y_test_path))

    #-----------------------------------
    # step2: define parameters for model
    #-----------------------------------
    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    #------------------------------------------------------
    # step3: Let's get serious and build the neural network
    #------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    #--------------------------------------------
    # step4: Hooray, now train the neural network
    #--------------------------------------------
    # Note that log_device_placement can be turned ON but will cause console spam.
    sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    tf.initialize_all_variables().run()

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
                                                X: X_test, Y: y_test})
        print("traing iter: {},".format(i)+\
              " test accuracy : {},".format(accuracy_out)+\
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")

    #------------------------------------------------------------------
    # step5: Training is good, but having visual insight is even better
    #------------------------------------------------------------------
    # The code is in the .ipynb

    #------------------------------------------------------------------
    # step6: And finally, the multi-class confusion matrix and metrics!
    #------------------------------------------------------------------
    # Input Data

    training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
    test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep


    # LSTM Neural Network's internal structure

    n_hidden = 32 # Hidden layer num of features
    n_classes = 6 # Total classes (should go up, or should go down)


    # Training

    learning_rate = 0.0025
    lambda_loss_amount = 0.0015
    training_iters = training_data_count * 300  # Loop 300 times on the dataset
    batch_size = 1500
    display_iter = 30000  # To show test set accuracy during training

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    ) # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) + l2 # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.initialize_all_variables()
    sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    while step * batch_size <= training_iters:
        batch_xs =         extract_batch_size(X_train, step, batch_size)
        batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs,
                y: batch_ys
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):

            # To not spam console, show training accuracy/loss in this "if"
            print "Training iter #" + str(step*batch_size) + \
                  ":   Batch Loss = " + "{:.6f}".format(loss) + \
                  ", Accuracy = {}".format(acc)

            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc = sess.run(
                [cost, accuracy],
                feed_dict={
                    x: X_test,
                    y: one_hot(y_test)
                }
            )
            test_losses.append(loss)
            test_accuracies.append(acc)
            print "PERFORMANCE ON TEST SET: " + \
                  "Batch Loss = {}".format(loss) + \
                  ", Accuracy = {}".format(acc)

        step += 1

    print "Optimization Finished!"


    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: X_test,
            y: one_hot(y_test)
        }
    )

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)

    print "FINAL RESULT: " + \
          "Batch Loss = {}".format(final_loss) + \
          ", Accuracy = {}".format(accuracy)


    # (Inline plots: )

    font = {
        'family' : 'Bitstream Vera Sans',
        'weight' : 'bold',
        'size'   : 18
    }
    matplotlib.rc('font', **font)

    width = 12
    height = 12
    plt.figure(figsize=(width, height))

    indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
    plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
    plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

    indep_test_axis = np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1] + [training_iters])
    plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
    plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper right', shadow=True)
    plt.ylabel('Training Progress (Loss or Accuracy values)')
    plt.xlabel('Training iteration')

    plt.show()


    predictions = one_hot_predictions.argmax(1)

    print "Testing Accuracy: {}%".format(100*accuracy)

    print ""
    print "Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted"))
    print "Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted"))
    print "f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted"))

    print ""
    print "Confusion Matrix:"
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)
    print confusion_matrix
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

    print ""
    print "Confusion matrix (normalised to % of total test data):"
    print normalised_confusion_matrix
    print ("Note: training and testing data is not equally distributed amongst classes, "
           "so it is normal that more than a 6th of the data is correctly classifier in the last category.")

    # Plot Results:
    width = 12
    height = 12
    plt.figure(figsize=(width, height))
    plt.imshow(
        normalised_confusion_matrix,
        interpolation='nearest',
        cmap=plt.cm.rainbow
    )
    plt.title("Confusion matrix \n(normalised to % of total test data)")
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
