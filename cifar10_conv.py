# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 20:03:50 2017

@author: rojod
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
import utils.utils as ut

import os

from keras.models import Sequential
from keras.datasets import cifar10

#from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from keras.layers import *
from keras.regularizers import l2


# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


X_train=x_train
X_test=x_test




#y
Y_train = to_categorical(y_train, 10)
Y_test = to_categorical(y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print( X_train.shape)

input1 = Input(shape=(32, 32, 3))


model = Sequential()
model.add(keras.layers.Convolution2D(
                                        36,
                                        kernel_size=(5,5),
                                        strides=(1,1),
                                        padding="same",
                                        dilation_rate=1,
                                        activation=None,
                                        input_shape=X_train.shape[1:]))
model.add(Activation("relu"))
#model.add(keras.layers.Convolution2D(
#                                        64,
#                                        kernel_size=(5,5),
#                                       strides=(1,1),
#                                        padding="same",
#                                        dilation_rate=2))
#model.add(Activation("relu"))
model.add(keras.layers.MaxPooling2D(
                                    (2,2),
                                    strides=None,
                                    padding="valid"))
model.add(Dropout(0.25))
model.add(keras.layers.Convolution2D(
                                        64,
                                        kernel_size=(3,3),
                                        strides=(1,1),
                                        padding="same",
                                        dilation_rate=2))
model.add(Activation("relu"))
#model.add(keras.layers.Convolution2D(
#                                        64,
#                                        kernel_size=(3,3),
#                                        strides=(1,1),
#                                        padding="same",
#                                        dilation_rate=2))
#model.add(Activation("relu"))
model.add(keras.layers.MaxPooling2D(
                                    (2,2),
                                    strides=None,
                                    padding="valid"))
model.add(Dropout(0.5))

model.add(Flatten())                                        
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Activation("relu"))
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation('softmax'))


print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=1e-3),
    metrics=['accuracy'])

fit = model.fit(
    X_train, Y_train,
    batch_size=100,
    epochs=1,
    verbose=2,
    validation_split=0.1,  # split off 10% training data for validation
    callbacks=[])


# ----------------------------------------------
# Some plots
# ----------------------------------------------

# predicted probabilities for the test set
Yp = model.predict(X_test)
yp = np.argmax(Yp, axis=1)

folder = 'results/cifar10/'
if not os.path.exists(folder):
    os.makedirs(folder)


cifar10_classes = np.array([
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'])

# plot some test images along with the prediction
'''
for i in range(10):
    ut.plot_prediction(
        Yp[i],
        data.test.images[i],
        data.test.labels[i],
        classes,
        fname=folder + 'test-%i.png' % i)
'''

print(yp, y_test)
# plot the confusion matrix
ut.plot_confusion(yp, y_test[:,0], cifar10_classes,
                           fname=folder + 'confusion.png')
                           


