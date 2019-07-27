import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, CuDNNLSTM, LSTM, Activation, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

print("Using TensorFlow version " + tf.__version__) # display version of tensorflow being used

mnist = keras.datasets.mnist
(train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = fashion_mnist.load_data() # load the fashion mnist dataset

l1_nodes = 500 # 500 nodes in l1
l2_nodes = 500
l3_nodes = 500
n_classes = 10 # numbers from 0-9 1 class for each number
batch_size = 32
epochs = 10 # number of times to run through the data

x = tf.compat.v1.placeholder(tf.float32, [None, 784]) # placeholder tensor for the flattened images
y = tf.compat.v1.placeholder(tf.float32) # placeholder tensor for the target one hot vector


X_train = [] # empty list of image arrays
for image in train_images_mnist:
    X_train.append(image.flatten()) # flatten 28x28 into 1x784
X_train = np.array(X_train)/255 # dived greyscale values by 255 to scale values between 0-1

X_test = []
for image in test_images_mnist:
    X_test.append(image.flatten())
X_test = np.array(X_test)/255

y_train = []
for num in train_labels_mnist:
    one_hot = np.zeros(10) # create array of zeros
    one_hot[num] = 1 # assign 1 in index of value to create a one hot array
    y_train.append(one_hot) # add one hot array to y_train
y_train = np.array(y_train) # cast list to numpy array

y_test = []
for num in test_labels_mnist:
    one_hot = np.zeros(10)
    one_hot[num] = 1
    y_test.append(one_hot)
y_test = np.array(y_test)


def NN(data):
    hidden_l1 = {'weights': tf.Variable(tf.random.truncated_normal([784, l1_nodes], stddev=0.1)), # create a 784xl1_nodes truncated normal variable with standard deviation of 0.1
                    'bias': tf.Variable(tf.constant(0.1, shape=[l1_nodes]))} # constant variable starting with values of all 0.1

    hidden_l2 = {'weights': tf.Variable(tf.random.truncated_normal([l1_nodes, l2_nodes], stddev=0.1)),
                    'bias': tf.Variable(tf.constant(0.1, shape=[l2_nodes]))}

    hidden_l3 = {'weights': tf.Variable(tf.random.truncated_normal([l2_nodes, l3_nodes], stddev=0.1)),
                    'bias': tf.Variable(tf.constant(0.1, shape=[l3_nodes]))}

    output = {'weights': tf.Variable(tf.random.truncated_normal([l3_nodes, n_classes], stddev=0.1)),
                    'bias': tf.Variable(tf.constant(0.1, shape=[n_classes]))}


    l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['bias']) # matrix multiply input data with weights and add bias vector
    l1 = tf.nn.relu(l1) # apply rectified linear activation function

    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['bias'])
    l3 = tf.nn.relu(l3)

    out = tf.add(tf.matmul(l3, output['weights']), output['bias'])

    return out

def train_NN(X):
    # note: the following few lines of code are not excecuted as eager excecution is not the default in this document

    pred = NN(X) # run the model

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y)) # calculate the cost function
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.03).minimize(cost) # optimizer to minimize cost

    # excecution of code starts in the session
    with tf.compat.v1.Session() as sess: # create session named sess
        sess.run(tf.compat.v1.global_variables_initializer()) # initialze all variables

        for epoch in range(epochs): # run through data epoch number of times
            epoch_loss = 0 # start the loss as 0

            for i in range(int(len(X_train)/batch_size)): # for i in number of batches of batch size batch_size
                lower = i * batch_size # start point of batch i
                upper = lower + batch_size # upper point of batch i
                epoch_x, epoch_y = X_train[lower:upper], y_train[lower:upper] # split data into the batch_size

                # backpropagation stage, optimizer changes tf.Variable(...) variables to reduce constant
                _, c = sess.run([optimizer, cost], {x: epoch_x, y: epoch_y}) # run optimizer and cost with the split of the batches

                epoch_loss += c # increase the loss by the calculated cost

            print('epoch: ', epoch + 1, '/', epochs, ' loss: ', epoch_loss) # print out epoch number and the cost at the end of the runthrough
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # do the predictions and true values match
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) # what proportion of the correct tensor that match
        print('Accuracy: ', accuracy.eval({x: X_test, y: y_test})) # evaluate the accuracy using the test split

train_NN(x)
