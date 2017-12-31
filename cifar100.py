# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 20:42:34 2017

@author: rojod
"""

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
from keras.datasets import cifar100

#from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.layers import *
from keras.regularizers import l2


# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------
(x_train, y_train), (x_test, y_test) = cifar100.load_data()


X_train=x_train
X_test=x_test



print( y_train.shape)
#y
Y_train = to_categorical(y_train, 100)
Y_test = to_categorical(y_test, 100)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train -= np.mean(X_train, axis = 0)
X_test -= np.mean(X_test, axis = 0)

X_train /= 255.
X_test /= 255.

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

z = Conv2D(32, (3, 3),activation='elu')( input1 )
Dropout(0.75)(z0)
z = Conv2D(32, (2, 2),activation='elu')( z )    
Dropout(0.75)(z0)
z = MaxPooling2D((2, 2), strides=(2, 2))(z)
z = Conv2D(32, (2, 2),activation='elu')( z )    
Dropout(0.75)(z)

z=residual_unit(z,32,drop=0.75)
    
z = MaxPooling2D((2, 2), strides=(1, 1))(z)
    
z=residual_unit(z,32,drop=0.75)
z = MaxPooling2D((2, 2), strides=(1, 1))(z)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z)

z1=z
z=residual_unit(z,32,drop=0.75)

BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z)
#z = MaxPooling2D((2, 2), strides=(1, 1))(z)
z=concatenate([z1, z])
z=Activation("elu")(z)

  
Dropout(0.75)(z)
z=residual_unit(z,32,drop=0.75)

z = MaxPooling2D((4, 4), strides=(2, 2))(z)
z=Flatten()(z)

output=Dense(100, activation= "softmax")(z)


model = keras.models.Model(inputs=input1, outputs=output)

print(model.summary())

    
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(lr=0.001),  #hyperparameter
    metrics=['accuracy'])

fit = model.fit(
    X_train, Y_train,
    batch_size=120,    #hyperparameter
    epochs=30,
    verbose=2,
    validation_split=0.15,  # split off 10% training data for validation
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=2. / 3, patience=10, verbose=1),
              EarlyStopping(monitor='val_loss', patience=15)])

# ----------------------------------------------
# Some plots
# ----------------------------------------------

# predicted probabilities for the test set
    
score = model.evaluate(X_test, Y_test, verbose=0)    
    
Yp = model.predict(X_test)
yp = np.argmax(Yp, axis=1)

folder = 'results/cifar100/'
if not os.path.exists(folder):
    os.makedirs(folder)
    
"""
wrong order...
cifar100_classes = np.array(["beaver", "dolphin", "otter", "seal", "whale",
                            "aquarium fish", "flatfish", "ray", "shark", "trout",
                            "orchids", "poppies", "roses", "sunflowers", "tulips",
                            "bottles", "bowls", "cans", "cups", "plates",
                            "apples", "mushrooms", "oranges", "pears", "sweet peppers",
                            "clock", "computer keyboard", "lamp", "telephone", "television",
                            "bed", "chair", "couch", "table", "wardrobe",
                            "bee", "beetle", "butterfly", "caterpillar", "cockroach",
                            "bear", "leopard", "lion", "tiger", "wolf",
                            "bridge", "castle", "house", "road", "skyscraper",
                            "cloud", "forest", "mountain", "plain", "sea",
                            "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
                            "fox", "porcupine", "possum", "raccoon", "skunk",
                            "crab", "lobster", "snail", "spider", "worm",
                            "baby", "boy", "girl", "man", "woman",
                            "crocodile", "dinosaur", "lizard", "snake", "turtle", 
                            "hamster", "mouse", "rabbit", "shrew", "squirrel",
                            "maple", "oak", "palm", "pine", "willow",
                            "bicycle", "bus", "motorcycle", "pickup truck", "train",
                            "lawn-mower", "rocket", "streetcar", "tank", "tractor"])
"""

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
for i in range(50):
    ut.plot_prediction(
        Yp[i],
        x_test[i],
        y_test[i],
        fname=folder + 'test-%i.png' % i,
        top_n=10)
print(yp, y_test)
# plot the confusion matrix
ut.plot_confusion(yp, y_test[:,0],
                           fname=folder + 'confusion.png')


good = np.count_nonzero(yp-y_test == 0)

print(good*1./yp.size) 

print(score)


