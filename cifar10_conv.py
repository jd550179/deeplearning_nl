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
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import *
from keras.regularizers import l2


# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


X_train=x_train
X_test=x_test


X_train, X_valid = np.split(x_train, [-7500])
y_train, y_valid = np.split(y_train, [-7500])






X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# channel-wise standard normalization
mX = np.mean(x_train, axis=(0, 1, 2))
sX = np.std(x_train, axis=(0, 1, 2))


X_train = (X_train - mX) / sX
X_valid = (X_valid - mX) / sX
X_test = (X_test - mX) / sX



print( y_train.shape)
#y
Y_train = to_categorical(y_train, 10)
Y_valid = to_categorical(y_valid, 10)
Y_test = to_categorical(y_test, 10)


print( X_train.shape)

input1 = Input(shape=(32, 32, 3))

###############################################################
#residual unit, the holy grail :)#
def residual_unit(z0, n,drop=0.0):
    
    zstart = Conv2D(n, (1, 1),padding="same",activation='elu')(z0)
    Dropout(drop)(zstart)
    BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(zstart)
    
    z = Conv2D(n, (3, 3),padding="same",activation='elu')(zstart)
    Dropout(drop)(z)
    BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z)
    z = Conv2D(n, (3, 3),padding="same")(z)
    Dropout(drop)(z)
    z = add([z,zstart])
    Dropout(drop)(z)
    return Activation("elu")(z)
###########################################################

z0 = Conv2D(32, (3, 3),activation='elu')( input1 )
Dropout(0.75)(z0)
z0 = Conv2D(32, (2, 2),activation='elu')( z0 )    
Dropout(0.75)(z0)


#PATH 1
z = MaxPooling2D((2, 2), strides=(2, 2))(z0)
z = Conv2D(32, (2, 2),activation='elu')( z )    
Dropout(0.75)(z)
z=residual_unit(z,32,drop=0.75)
z = MaxPooling2D((2, 2), strides=(1, 1))(z) 
z=residual_unit(z,32,drop=0.75)
z = MaxPooling2D((2, 2), strides=(1, 1))(z)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z)
z=residual_unit(z,32,drop=0.75)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z)


#PATH 2
z1= Conv2D(32, (2, 2),activation='elu')( z0 )
z1= Conv2D(32, (2, 2),activation='elu')( z1 )
z1 = MaxPooling2D((4, 4), strides=(2, 2))(z1) 
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z1)
z1= Conv2D(32, (2, 2),activation='elu')( z1 )







#Concatenate them
z=concatenate([z, z1])
  
Dropout(0.75)(z)
z=residual_unit(z,128,drop=0.75)

z = MaxPooling2D((4, 4), strides=(2, 2))(z)
z=Flatten()(z)

output=Dense(10, activation= "softmax")(z)


model = keras.models.Model(inputs=input1, outputs=output)
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=["accuracy"])

# data augmentation, what does this do?
generator = ImageDataGenerator(
    width_shift_range=4. / 32,
    height_shift_range=4. / 32,
    fill_mode='constant',
    horizontal_flip=True,
    rotation_range=5.)

batch_size = 50
steps_per_epoch = len(X_train) // batch_size

# fit using augmented data generator
fit=model.fit_generator(
    generator.flow(X_train, Y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=(X_valid, Y_valid),  # fit_generator doesn't work with validation_split
    verbose=2,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=2. / 3, patience=5, verbose=1),
               EarlyStopping(monitor='val_loss', patience=10)])



# ----------------------------------------------
# Some plots
# ----------------------------------------------

# predicted probabilities for the test set
    
[loss, accuracy] = model.evaluate(X_test, Y_test, verbose=0) 
    
Yp = model.predict(X_test)
yp = np.argmax(Yp, axis=1)

folder = 'results/cifar10/'
if not os.path.exists(folder):
    os.makedirs(folder)
    

# plot some test images along with the prediction

for i in range(10):
    ut.plot_prediction(
        Yp[i],
        x_test[i],
        y_test[i],
        fname=folder + 'test-%i.png' % i,
        top_n=10)
print(yp, y_test)

print("loss:", loss)

print("accuracy:", accuracy)


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



print(yp, y_test)
# plot the confusion matrix
ut.plot_confusion(yp, y_test[:,0], cifar10_classes,
                           fname=folder + 'confusion.png')
                           


