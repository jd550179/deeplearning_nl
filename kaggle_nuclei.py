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
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_true = tf.to_int32(y_true)
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)
'''
def read_img(img_id, train_or_test, size, mask="images"):
    img = image.load_img(data_dir + train_or_test +"/"+str(img_id)+"/images/"+str(img_id)+".png", target_size=(size, size))
    img = image.img_to_array(img)
    return img

def read_mask(img_id, size):
    directory=data_dir + "stage1_train/"+str(img_id)+"/masks/"
    maskfiles= os.listdir(directory)
    mask_array=np.zeros((size,size,3))
    for j in (maskfiles):
        img = image.load_img(directory+str(j), target_size=(size, size))
        img = image.img_to_array(img)
        mask_array=np.add(mask_array,img)
    return mask_array[:,:,0].reshape((size,size,1))


#@vectorize(["float32(string,int32,int32)"] , target = 'cpu')
def read_data(img_id, train, size):
    progress=0
    if train == 1:
        directory="stage1_train"
        mask_train_list=[]

    else:
        directory="stage1_test"
    X_train_list=[]
    unique, indices=np.unique(img_id, return_index=True)
    for i in (unique):
        if progress%50==0:
            print( progress)
        img = read_img(str(i), directory, size)/255.
        X_train_list.append(img)

        if train==1:
             mask_train_list.append((read_mask(i, size)/255))
        progress+=1
    if train==1:
        return np.array(X_train_list), np.array(mask_train_list)
    else:
        return np.array(X_train_list)
'''
x_train, y_train=read_data(labels.ImageId.values, 1,  size)


folder = 'results/nuclei/'
if not os.path.exists(folder):
    os.makedirs(folder)

print(x_train.shape)
print(y_train.shape)
print(np.unique(y_train))



#X_train, X_valid = np.split(x_train, [-75])
#Y_train, Y_valid = np.split(y_train, [-75])

X_train = x_train.astype('float32')
#X_valid = x_valid.astype('float32')
#X_test = X_test.astype('float32')
# channel-wise standard normalization
mX = np.mean(X_train, axis=(0, 1, 2))
sX = np.std(X_train, axis=(0, 1, 2))

X_train = (X_train - mX) / sX
#X_valid = (X_valid - mX) / sX

Y_train=y_train


input1 = Input(shape=(size, size, 3))

z0=Conv2D(8, (4,4), activation="relu", padding="same")(input1)
Dropout(0.25)(z0)

z1=MaxPooling2D((2, 2), strides=(2, 2))(z0)


z1_1=Conv2D(8, (4,4), activation="relu", padding="same")(z1)
Dropout(0.25)(z1_1)
z1_2=concatenate([z1_1,z1])  #size 16
z1_3=Conv2D(16, (4,4), activation="relu", padding="same")(z1_2)
Dropout(0.25)(z1_3)
z1_4=concatenate([z1_3,z1_2])  #size 32
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z1_4)
z2=MaxPooling2D((2, 2), strides=(2, 2))(z1_4)

z2_1=Conv2D(16, (4,4), activation="relu", padding="same")(z2)
Dropout(0.25)(z2_1)
z2_2=concatenate([z2_1,z2]) #size 48
z2_3=Conv2D(16, (4,4), activation="relu", padding="same")(z2_2)
Dropout(0.25)(z2_3)
z2_4=concatenate([z2_3,z2_2])  #size 64
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z2_4)
z3=MaxPooling2D((2, 2), strides=(2, 2))(z2_4)

z3_1=Conv2D(32, (4,4), activation="relu", padding="same")(z3)
Dropout(0.25)(z3_1)
z3_2=concatenate([z3_1,z3])  #size 96
z3_3=Conv2D(32, (2,2), activation="relu", padding="same")(z3_2)
Dropout(0.25)(z3_3)
z3_4=concatenate([z3_3,z3_2])   #size 128
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z3_4)
z4=MaxPooling2D((2, 2), strides=(2, 2))(z3_4)


z4_1=Conv2D(64, (2,2), activation="relu", padding="same")(z4)
Dropout(0.25)(z4_1)
z4_2=concatenate([z4_1,z4])  #size 192
z4_3=Conv2D(64, (2,2), activation="relu", padding="same")(z4_2)
Dropout(0.25)(z4_3)
z4_4=concatenate([z4_3,z4_2])  #size 256
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z4_4)
z5=MaxPooling2D((2, 2), strides=(2, 2))(z4_4)

z5_1=Conv2D(128, (2,2), activation="relu", padding="same")(z5)
Dropout(0.25)(z5_1)
z5_2=concatenate([z5_1,z5])  #size 384
z5_3=Conv2D(128, (2,2), activation="relu", padding="same")(z5_2)
Dropout(0.25)(z5_3)
z5_4=concatenate([z5_3,z5_2])  #size 512
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z5_4)
z6=MaxPooling2D((2, 2), strides=(2, 2))(z5_4)



z6_1=Conv2D(256, (2,2), activation="relu", padding="same")(z6)
Dropout(0.25)(z6_1)
z6_2=concatenate([z6_1,z6])   #size 768
z6_3=Conv2D(256, (1,1), activation="relu", padding="same")(z6_2)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(z6_3)
Dropout(0.25)(z6_3)
z6_4=concatenate([z6_3,z6_2])  #size 1024



u5=UpSampling2D((2, 2))(z6_4)

u5_1=Conv2D(512, (2,2), activation="relu", padding="same")(u5)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(u5_1)
u5_2=add([u5_1,z5_4])
u4=UpSampling2D((2, 2))(u5_2)

u4_1=Conv2D(256, (2,2), activation="relu", padding="same")(u4)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(u4_1)
u4_2=add([u4_1,z4_4])
u3=UpSampling2D((2, 2))(u4_2)

u3_1=Conv2D(128, (2,2), activation="relu", padding="same")(u3)
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(u3_1)
u3_2=add([u3_1,z3_4])
u2=UpSampling2D((2, 2))(u3_2)

u2_1=Conv2D(64, (4,4), activation="relu", padding="same")(u2)
u2_2=add([u2_1,z2_4])
BatchNormalization(gamma_regularizer=l2(1E-4),
                           beta_regularizer=l2(1E-4))(u2_2)
u1=UpSampling2D((2, 2))(u2_2)

u1_1=Conv2D(32, (4,4), activation="relu", padding="same")(u1)
u1_2=add([u1_1,z1_4])
u0=UpSampling2D((2, 2))(u1_2)

u=Conv2D(8, (2,2), activation="relu", padding="same")(u0)
u_last=concatenate([u,z0])
output=Conv2D(1, (1,1), activation="sigmoid", padding="same")(u_last)

model = keras.models.Model(inputs=input1, outputs=output)
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=[mean_iou])

# data augmentation, what does this do?

generator = ImageDataGenerator()

batch_size = 16
#steps_per_epoch = len(X_train) // batch_size
checkpointer = ModelCheckpoint('model_dsbowl2018_1.h5', verbose=1, save_best_only=True)

# fit using augmented data generator
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
fit=model.fit_generator(
    generator.flow(X_train, Y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=100,
    validation_data=(X_valid, Y_valid),  # fit_generator doesn't work with validation_split
    verbose=2,
    callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=2. / 3, patience=5, verbose=1),
               EarlyStopping(monitor='val_loss', patience=10)])
'''
Yp_train = model.predict(X_train)

plt.imshow(Yp_train[0,:,:,0])
plt.savefig(folder+"photos/train_output0.png")
