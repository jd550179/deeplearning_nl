# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:19:01 2017

@author: rojod
"""
import numpy as np
import matplotlib.pyplot as plt
import os

import utils.utils

from tensorflow.examples.tutorials.mnist import input_data


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical


data = input_data.read_data_sets("MNIST_data/")



# reshape the image matrices to vectors
X_train = data.train.images.reshape(-1, 28**2)
X_test = data.test.images.reshape(-1, 28**2)
print('%i training samples' % X_train.shape[0])
print('%i test samples' % X_test.shape[0])

# convert integer RGB values (0-255) to float values (0-1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# convert class labels to one-hot encodings
Y_train = to_categorical(data.train.labels,10)
Y_test =to_categorical(data.test.labels,10)


# ----------------------------------------------
# Model and training
# ----------------------------------------------

# make output directory
folder = 'results/'
if not os.path.exists(folder):
    os.makedirs(folder)

model = Sequential([
    Dense(64, input_shape=(784,)),
    Activation('relu'),
    Dropout(0.5),
    Dense(10),
    Activation('softmax')])

print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=1e-3),
    metrics=['accuracy'])

fit = model.fit(
    X_train, Y_train,
    batch_size=100,
    epochs=10,
    verbose=2,
    validation_split=0.1,  # split off 10% training data for validation
    callbacks=[])


# ----------------------------------------------
# Some plots
# ----------------------------------------------

# predicted probabilities for the test set
Yp = model.predict(X_test)
yp = np.argmax(Yp, axis=1)

classes= np.array([0,1,2,3,4,5,6,7,8,9])

# plot some test images along with the prediction
for i in range(10):
    print(data.test.labels[i])
    utils.utils.plot_prediction(
        Yp[i],
        data.test.images[i],
        data.test.labels[i],
        classes,
        fname=folder + 'test-%i.png' % i)

# plot the confusion matrix
utils.utils.plot_confusion(yp, data.test.labels, classes,
                           fname=folder + 'confusion.png')
                           