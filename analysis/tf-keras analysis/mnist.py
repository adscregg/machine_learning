import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, CuDNNLSTM, LSTM, Activation, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

print("Using TensorFlow version " + tf.__version__)

mnist = keras.datasets.mnist
(train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = mnist.load_data()

l1_nodes = 500
l2_nodes = 500
l3_nodes = 500
n_classes = 10
batch_size = 32
epochs = 10

x = tf.compat.v1.placeholder(tf.float32, [None, 784])
y = tf.compat.v1.placeholder(tf.float32)


X_train = []
for i, im in enumerate(train_images_mnist):
    X_train.append(train_images_mnist[i].flatten())
X_train = np.array(X_train)/255

X_test = []
for i, im in enumerate(test_images_mnist):
    X_test.append(test_images_mnist[i].flatten())
X_test = np.array(X_test)/255

y_train = []
for i, num in enumerate(train_labels_mnist):
    one_hot = np.zeros(10)
    one_hot[num] = 1
    y_train.append(one_hot)
y_train = np.array(y_train)

y_test = []
for i, num in enumerate(test_labels_mnist):
    one_hot = np.zeros(10)
    one_hot[num] = 1
    y_test.append(one_hot)
y_test = np.array(y_test)


def NN(data):
    hidden_l1 = {'weights': tf.Variable(tf.random.truncated_normal([784, l1_nodes], stddev=0.1)),
                    'bias': tf.Variable(tf.constant(0.1, shape=[l1_nodes]))}

    hidden_l2 = {'weights': tf.Variable(tf.random.truncated_normal([l1_nodes, l2_nodes], stddev=0.1)),
                    'bias': tf.Variable(tf.constant(0.1, shape=[l2_nodes]))}

    hidden_l3 = {'weights': tf.Variable(tf.random.truncated_normal([l2_nodes, l3_nodes], stddev=0.1)),
                    'bias': tf.Variable(tf.constant(0.1, shape=[l3_nodes]))}

    output = {'weights': tf.Variable(tf.random.truncated_normal([l3_nodes, n_classes], stddev=0.1)),
                    'bias': tf.Variable(tf.constant(0.1, shape=[n_classes]))}


    l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['bias'])
    l3 = tf.nn.relu(l3)

    out = tf.add(tf.matmul(l3, output['weights']), output['bias'])

    return out

def train_NN(X):
    pred = NN(X)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.03).minimize(cost)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # File_Writer = tf.summary.FileWriter('C:\\Users\\Anthony\\Documents\\Python files\\graph', sess.graph)

        for epoch in range(epochs):
            epoch_loss = 0

            for i in range(int(len(X_train)/batch_size)):
                lower = i * batch_size
                upper = lower + batch_size
                epoch_x, epoch_y = X_train[lower:upper], y_train[lower:upper]

                _, c = sess.run([optimizer, cost], {x: epoch_x, y: epoch_y})

                epoch_loss += c

            print('epoch: ', epoch + 1, '/', epochs, ' loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        print('Accuracy: ', accuracy.eval({x: X_test, y: y_test}))

train_NN(x)
