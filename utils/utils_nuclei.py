# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import os

from keras.preprocessing import image
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy


from skimage.morphology import label
from skimage.transform import resize


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
    return K.mean(K.stack(prec))


def dice_coef(y_true, y_pred):
    smooth = 1.
    #y_true = tf.to_int32(y_true)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    #y_true = tf.to_int32(y_true)
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)



def read_img(img_id, train_or_test, size, data_dir):
    if size!=None:
        img = image.load_img(data_dir + train_or_test + str(img_id)+"/images/"+str(img_id)+".png", target_size=(size, size))
    else:
        img = image.load_img(data_dir + train_or_test + str(img_id)+"/images/"+str(img_id)+".png")
    img = image.img_to_array(img)
    return img

def read_mask(img_id, size, data_dir):
    directory=data_dir + "stage1_train/"+str(img_id)+"/masks/"
    maskfiles= os.listdir(directory)
    mask_array=np.zeros((size,size,3))
    for j in (maskfiles):
        img = image.load_img(directory+str(j), target_size=(size, size))
        img = image.img_to_array(img)
        mask_array=np.add(mask_array,img)
    return mask_array[:,:,0].reshape((size,size,1))

def read_data(img_id, train, size, data_dir):
    progress=0
    if train == 1:
        directory="stage1_train/"
        mask_train_list=[]

    else:
        directory="stage1_test/"
    X_train_list=[]
    unique, indices=np.unique(img_id, return_index=True)
    for i in (unique):
        if progress%50==0:
            print( progress)
        img = read_img(str(i), directory, size, data_dir)/255.
        X_train_list.append(img)

        if train==1:
             mask_train_list.append((read_mask(i, size, data_dir)/255))
        progress+=1
    if train==1:
        return np.array(X_train_list), np.array(mask_train_list)
    else:
        return np.array(X_train_list)



def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
