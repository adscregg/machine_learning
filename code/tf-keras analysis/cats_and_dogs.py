import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, CuDNNLSTM, LSTM, Activation, Flatten, MaxPool2D, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np


pets_dir = '../../datasets/cats_and_dogs_NN/PetImages'
categories = os.listdir(pets_dir)


def create_training_data(categories, data_dir, img_size):
    """Reads image files and processes them into the correct format to be read by a keras NN.

    Parameters
    ----------
    categories : array-like object
        List of folder name to be read from in the data_dir parameter.
    data_dir : string
        Filepath to the directory where the categories folders are stored.
    img_size : 2d-tuple
        The dimensions to reshape the image to (width, height).

    Returns
    -------
    tuple (X, y)
        X is the training array of images and y is the target for each X.

    """
    training_data = []
    for category in categories:
        target = categories.index(category)
        path = os.path.join(data_dir, category)
        for image in os.listdir(path):
            try:
                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                resized_image = cv2.resize(image_array, img_size)/255
                training_data.append([resized_image, target])
            except Exception as e:
                pass

    random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1)

    return X, y

X, y = create_training_data(categories, pets_dir, (75, 75))



model_pets = Sequential()

model_pets.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model_pets.add(Activation('relu'))
model_pets.add(MaxPool2D(pool_size=(2,2)))
model_pets.add(Dropout(0.25))

model_pets.add(Conv2D(64, (3,3)))
model_pets.add(Activation('relu'))
model_pets.add(MaxPool2D(pool_size=(2,2)))
model_pets.add(Dropout(0.3))

model_pets.add(Conv2D(64, (3,3)))
model_pets.add(Activation('relu'))
model_pets.add(MaxPool2D(pool_size=(2,2)))

model_pets.add(Flatten())
model_pets.add(Dense(64))
model_pets.add(Activation('relu'))

model_pets.add(Dense(1))
model_pets.add(Activation('sigmoid'))

model_pets.compile(loss='binary_crossentropy',
                   optimizer = 'adam',
                   metrics = ['accuracy'])

model_pets.fit(X, y, batch_size = 32, validation_split = 0.1, epochs = 5)
