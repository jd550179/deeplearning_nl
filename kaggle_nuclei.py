# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras
import tensorflow as tf
import keras.backend as K
import utils.utils as ut

from utils.utils_nuclei import *

import os

from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras.layers import *
from keras.regularizers import l2

from keras.preprocessing import image

#from numba import vectorize

data_dir="nuclei/"

labels = pd.read_csv(data_dir+ 'stage1_train_labels.csv')
sample_submission = pd.read_csv(data_dir + 'stage1_sample_submission.csv')

#labels_np= np.array(labels)
#print(labels.ImageId.values[0])
#print(labels.ImageId)

#print(labels.EncodedPixels.shape)

size=256


x_train, y_train=read_data(labels.ImageId.values, 1,  size, data_dir)


folder = 'results/nuclei/'
if not os.path.exists(folder):
    os.makedirs(folder)

print(x_train.shape)
print(y_train.shape)
print(np.unique(y_train))



X_train, X_valid = np.split(x_train, [-65])
Y_train, Y_valid = np.split(y_train, [-65])

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
#X_test = X_test.astype('float32')
# channel-wise standard normalization
mX = np.mean(X_train, axis=(0, 1, 2))
sX = np.std(X_train, axis=(0, 1, 2))

X_train = (X_train - mX) / sX
X_valid = (X_valid - mX) / sX

#Y_train=y_train

#############################
#Model
#############################

input1 = Input(shape=(size, size, 3))

z0=Conv2D(8, (4,4), activation="elu", padding="same")(input1)
Dropout(0.15)(z0)

z1=MaxPooling2D((2, 2), strides=(2, 2))(z0)


z1_1=Conv2D(8, (4,4), activation="elu", padding="same")(z1)
Dropout(0.15)(z1_1)
z1_2=concatenate([z1_1,z1])  #size 16
z1_3=Conv2D(16, (4,4), activation="elu", padding="same")(z1_2)
Dropout(0.15)(z1_3)
z1_4=concatenate([z1_3,z1_2])  #size 32
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z1_4)
z2=MaxPooling2D((2, 2), strides=(2, 2))(z1_4)

z2_1=Conv2D(16, (4,4), activation="elu", padding="same")(z2)
Dropout(0.15)(z2_1)
z2_2=concatenate([z2_1,z2]) #size 48
z2_3=Conv2D(16, (4,4), activation="elu", padding="same")(z2_2)
Dropout(0.15)(z2_3)
z2_4=concatenate([z2_3,z2_2])  #size 64
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z2_4)
z3=MaxPooling2D((2, 2), strides=(2, 2))(z2_4)

z3_1=Conv2D(32, (4,4), activation="elu", padding="same")(z3)
Dropout(0.15)(z3_1)
z3_2=concatenate([z3_1,z3])  #size 96
z3_3=Conv2D(32, (2,2), activation="elu", padding="same")(z3_2)
Dropout(0.15)(z3_3)
z3_4=concatenate([z3_3,z3_2])   #size 128
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z3_4)
z4=MaxPooling2D((2, 2), strides=(2, 2))(z3_4)


z4_1=Conv2D(64, (2,2), activation="elu", padding="same")(z4)
Dropout(0.15)(z4_1)
z4_2=concatenate([z4_1,z4])  #size 192
z4_3=Conv2D(64, (2,2), activation="elu", padding="same")(z4_2)
Dropout(0.15)(z4_3)
z4_4=concatenate([z4_3,z4_2])  #size 256
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z4_4)
z5=MaxPooling2D((2, 2), strides=(2, 2))(z4_4)

z5_1=Conv2D(128, (2,2), activation="elu", padding="same")(z5)
Dropout(0.15)(z5_1)
z5_2=concatenate([z5_1,z5])  #size 384
z5_3=Conv2D(128, (2,2), activation="elu", padding="same")(z5_2)
Dropout(0.15)(z5_3)
z5_4=concatenate([z5_3,z5_2])  #size 512
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z5_4)


u4=Conv2DTranspose(256, (2,2), strides=(2, 2), activation="linear", padding="same")(z5_4)


u4_1=Conv2D(256, (2,2), activation="elu", padding="same")(u4)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(u4_1)
u4_2=add([u4_1,z4_4])
Activation("elu")(u4_2)
u4_3=Conv2D(256, (2,2), activation="elu", padding="same")(u4_2)
Dropout(0.15)(u4_3)
u4_4=add([u4_2,u4_3])
Activation("elu")(u4_2)
u3=Conv2DTranspose(128, (2,2), strides=(2, 2), activation="linear", padding="same")(u4_4)


u3_1=Conv2D(128, (2,2), activation="elu", padding="same")(u3)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(u3_1)
u3_2=add([u3_1,z3_4])
Activation("elu")(u3_2)
u3_3=Conv2D(128, (2,2), activation="elu", padding="same")(u3_2)
Dropout(0.15)(u3_3)
u3_4=add([u3_2,u3_3])
Activation("elu")(u3_2)
u2=Conv2DTranspose(64, (2,2), strides=(2, 2), activation="linear", padding="same")(u3_4)


u2_1=Conv2D(64, (4,4), activation="elu", padding="same")(u2)
u2_2=add([u2_1,z2_4])
Activation("elu")(u2_2)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(u2_2)
u2_3=Conv2D(64, (4,4), activation="elu", padding="same")(u2_2)
Dropout(0.15)(u2_3)
u2_4=add([u2_2,u2_3])
Activation("elu")(u2_2)
u1=Conv2DTranspose(32, (2,2), strides=(2, 2), activation="linear", padding="same")(u2_4)


u1_1=Conv2D(32, (4,4), activation="elu", padding="same")(u1)
u1_2=add([u1_1,z1_4])
Activation("elu")(u1_2)
u0=Conv2DTranspose(8, (2,2), strides=(2, 2), activation="linear", padding="same")(u1_2)


u=Conv2D(8, (2,2), activation="elu", padding="same")(u0)
u_last=concatenate([u,z0])
output=Conv2D(3, (1,1), activation="softmax", padding="same")(u_last)  #softmax???

###############################################################################

model = keras.models.Model(inputs=input1, outputs=output)
model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=[mean_iou])

batch_size = 16
steps_per_epoch = len(X_train) // batch_size
checkpointer = ModelCheckpoint('model_dsbowl2018_4.h5', verbose=1, save_best_only=True)

# data augmentation, what does this do?
image_datagen = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)

mask_datagen = ImageDataGenerator(horizontal_flip=True,
                         vertical_flip=True,
                         rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1)
image_datagen.fit(X_train, seed=7)
mask_datagen.fit(Y_train, seed=7)
image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=7)
mask_generator = mask_datagen.flow(Y_train, batch_size=batch_size, seed=7)
train_generator = zip(image_generator, mask_generator)



'''
fit = model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=100,
    verbose=2,
    validation_split=0.1,  # split off 10% training data for validation
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=2. / 3, patience=5, verbose=1),
               EarlyStopping(monitor='val_loss', patience=15),
               checkpointer])
'''
# fit using augmented data generator
fit=model.fit_generator(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=200,
    validation_data=(X_valid, Y_valid),  # fit_generator doesn't work with validation_split
    verbose=2,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=2. / 3, patience=5, verbose=1),
               EarlyStopping(monitor='val_loss', patience=15),
               checkpointer])



Yp_train = model.predict(X_train)
if not os.path.exists(folder+"photos/train/"):
    os.makedirs(folder+"photos/train/")

for i in range(10):
    plt.imshow(Yp_train[i,:,:,0])
    plt.savefig(folder+"photos/train/output"+str(i)+".png")

    plt.imshow(x_train[i])
    plt.savefig(folder+"photos/train/image"+str(i)+".png")

    plt.imshow(Y_train[i,:,:,0])
    plt.savefig(folder+"photos/train/mask"+str(i)+".png")
