import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, CuDNNLSTM, LSTM, Activation, Flatten, MaxPooling2D
import matplotlib.pyplot as plt

print("Using TensorFlow version " + tf.__version__)

mnist = keras.datasets.mnist
(train_images_mnist, train_labels_mnist), (test_images_mnist, test_labels_mnist) = mnist.load_data()
