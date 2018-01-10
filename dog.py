# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 16:11:18 2017

@author: rojod
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
import utils.utils as ut

import os

from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import *
from keras.regularizers import l2

from keras.preprocessing import image

#from numba import vectorize

data_dir="doggos/"

labels = pd.read_csv(data_dir+ 'labels.csv')
sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')

#labels_np= np.array(labels)
print(labels.id.values[0])

size=128

def read_img(img_id, train_or_test, size):
    img = image.load_img(data_dir + train_or_test +"/"+str(img_id)+".jpg", target_size=(size, size))
    img = image.img_to_array(img)
    return img

#@vectorize(["float32(string,int32,int32)"] , target = 'cpu')
def read_data(img_id, train, size):
    if train == 1:
        directory="train"
    else: 
        directory="test"
    X_train_list=[]
    print(img_id.size)
    for i in range(int(img_id.size)):
        if i%1000==0:
            print( i)
        img = read_img(img_id[i], directory, size)/255.
        X_train_list.append(img)
        
    return np.array(X_train_list)


x_train=read_data(labels.id.values, 1,  size)

print(x_train.shape)


y_train_labels=labels.breed.values
b, c = np.unique(y_train_labels, return_inverse=True)






X_train, X_valid = np.split(x_train, [-1200])
y_train, y_valid = np.split(c, [-1200])






X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
#X_test = X_test.astype('float32')
# channel-wise standard normalization
mX = np.mean(x_train, axis=(0, 1, 2))
sX = np.std(x_train, axis=(0, 1, 2))


X_train = (X_train - mX) / sX
X_valid = (X_valid - mX) / sX
#X_test = (X_test - mX) / sX



print( y_train.shape)
#y
Y_train = to_categorical(y_train, 120)
Y_valid = to_categorical(y_valid, 120)
#Y_test = to_categorical(y_test, 10)

print (X_train.shape)








input1 = Input(shape=(size, size, 3))


#print(X_Train.shape)

def residual_unit(z0, n,drop=0.0):
    
    zstart = Conv2D(n, (1, 1),padding="same",activation='relu')(z0)
    Dropout(drop)(zstart)
    BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(zstart)
    
    z = Conv2D(n, (4, 4),padding="same",activation='relu')(zstart)
    Dropout(drop)(z)
    BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z)
    z = Conv2D(n, (4, 4),padding="same",activation='relu')(z)
    Dropout(drop)(z)
    z = add([z,zstart])
    Dropout(drop)(z)
    return Activation("relu")(z)


z0 = Conv2D(16, (5, 5),activation='relu')( input1 )
    
z = MaxPooling2D((4, 4), strides=(2, 2))(z0)


z0 = Conv2D(16, (5, 5),activation='relu')( z0 )
    
z0 = Conv2D(16, (5, 5),activation='relu')( z0 )
z0 = MaxPooling2D((5, 5), strides=(2, 2))(z0)



z=residual_unit(z0,16,drop=0.25)
z = MaxPooling2D((4, 4), strides=(2, 2))(z)
z=residual_unit(z,16,drop=0.5)
z=residual_unit(z,32,drop=0.5)
z=residual_unit(z,32,drop=0.5)
z=residual_unit(z,64,drop=0.5)

z1= Conv2D(32, (2, 2), padding= "same", activation='elu')( z0 )
z1= Conv2D(32, (2, 2), padding= "same", activation='elu')( z1 )
z1 = MaxPooling2D((4, 4), strides=(2, 2))(z1) 
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z1)
z1= Conv2D(32, (2, 2), padding= "same", activation='elu')( z1 )

z=concatenate([z, z1])


z2=residual_unit(z,32,drop=0.5)

z=concatenate([z, z2])

BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z)
z=Flatten()(z)                                   
       
output = Dense(120, activation="softmax" )(z)    
    
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
    

'''
folder = 'residual/results/'
if not os.path.exists(folder):
    os.makedirs(folder)


Yp = model.predict(X_test)
yp = np.argmax(Yp, axis=1)

# plot the confusion matrix
dlipr.utils.plot_confusion(yp, data.test_labels, data.classes,
                              fname=folder + 'confusion.png')

for i in range(3):
    dlipr.utils.plot_prediction(
        Yp[i],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname=folder + 'test-%i.png' % i)
 

[loss, accuracy] = model.evaluate(X_test, Y_test, verbose=0)

vall_loss=fit.history["val_loss"][-1]
vall_accuracy=fit.history["val_acc"][-1]



print ("testloss:", loss)
print ("testaccuracy:", accuracy)
print ("vall loss:", vall_loss)
print ("vall accuracy:", vall_accuracy)

'''