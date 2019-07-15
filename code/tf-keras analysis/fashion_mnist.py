import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, CuDNNLSTM, LSTM, Activation, Flatten, MaxPooling2D
import matplotlib.pyplot as plt

print("Using TensorFlow version " + tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images_fashion, train_labels_fashion), (test_images_fashion, test_labels_fashion) = fashion_mnist.load_data()
