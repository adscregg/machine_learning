import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, CuDNNLSTM, LSTM, Activation, Flatten, MaxPool2D, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import os
import cv2
import random
import numpy as np



pets_dir = '../../datasets/cats_and_dogs_NN/PetImages' # location of the images directory (local file path)
categories = os.listdir(pets_dir) # list all files in the above directory


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
    training_data = [] # will contain all training images and targets
    for category in categories: # loop through all categories
        target = categories.index(category) # get index of category as the target
        path = os.path.join(data_dir, category) # add the name of the category (folder) to the file path
        for image in os.listdir(path): # iterate over the images in this directory
            try: # try to do the following, if fails excecute except statement
                image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE) # read the image into greyscale
                resized_image = cv2.resize(image_array, img_size)/255 # resize the image to the specified size and divide all elemnts by 255 so it can be trained on easier as algorthims prefer 0-1 values
                training_data.append([resized_image, target]) # add the resized array and the target to the training_data list
            except Exception as e:
                pass # do nothing if fail to read image

    random.shuffle(training_data) # shuffle the data inplace

    X = [] # empty list of images
    y = [] # empty list of targets

    for features, label in training_data:
        X.append(features) # append image to X
        y.append(label) # append target to y

    X = np.array(X).reshape(-1, img_size[0], img_size[1], 1) # keras network requires 4D inputs (n_samples, height, width, colour channels)

    return X, y

X, y = create_training_data(categories, pets_dir, (75, 75))



model_pets = Sequential() # keras Sequential model, stack of layers

model_pets.add(Conv2D(64, (3,3), input_shape = X.shape[1:])) # 2D convolutional layer with a 3x3 filter
model_pets.add(Activation('relu')) # rectified linear activation function
model_pets.add(MaxPool2D(pool_size=(2,2))) # maxpooling on 2x2 'chuncks' of convolved imaged
model_pets.add(Dropout(0.25)) #

model_pets.add(Conv2D(64, (3,3)))
model_pets.add(Activation('relu'))
model_pets.add(MaxPool2D(pool_size=(2,2)))
model_pets.add(Dropout(0.3))

model_pets.add(Conv2D(64, (3,3)))
model_pets.add(Activation('relu'))
model_pets.add(MaxPool2D(pool_size=(2,2)))

model_pets.add(Flatten()) # flatten the image into 1D array
model_pets.add(Dense(64)) # fully connected layer with 64 neurons
model_pets.add(Activation('relu'))

model_pets.add(Dense(1)) # single output neuron
model_pets.add(Activation('sigmoid')) # sigmoid activation

model_pets.compile(loss='binary_crossentropy',
                   optimizer = 'adam',
                   metrics = ['accuracy']) # metrics to track

model_pets.fit(X, y, batch_size = 32, validation_split = 0.1, epochs = 5)
