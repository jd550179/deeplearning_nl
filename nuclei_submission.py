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

from skimage.morphology import label
from skimage.transform import resize

from skimage.filters import sobel
from skimage.morphology import watershed

from scipy import ndimage as ndi


data_dir="nuclei/"

folder = 'results/nuclei/'
if not os.path.exists(folder):
    os.makedirs(folder)

labels = pd.read_csv(data_dir+ 'stage1_train_labels.csv')
sample_submission = pd.read_csv(data_dir + 'stage1_sample_submission.csv')

size=256

#model1=keras.models.load_model('model_dsbowl2018_1.h5', custom_objects={'mean_iou': mean_iou})
#model2=keras.models.load_model('model_dsbowl2018_2.h5', custom_objects={'mean_iou': mean_iou, "bce_dice_loss": bce_dice_loss})
model3=keras.models.load_model('model_dsbowl2018_4.h5', custom_objects={'mean_iou': mean_iou})
#x_train, y_train=read_data(labels.ImageId.values, 1,  size, data_dir)
x_test=read_data(sample_submission.ImageId.values, 0,  size, data_dir)

x_test_original=read_data(sample_submission.ImageId.values, 0,  None, data_dir)

#X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
# channel-wise standard normalization
#mX = np.mean(X_train, axis=(0, 1, 2))
#sX = np.std(X_train, axis=(0, 1, 2))

#X_train = (X_train - mX) / sX


mX_test = np.mean(X_test, axis=(0, 1, 2))
sX_test = np.std(X_test, axis=(0, 1, 2))

X_test = (X_test - mX_test) / sX_test




#Y_train=y_train

#Yp_train = model.predict(X_train)
Yp_raw = (model3.predict(X_test))# + model2.predict(X_test) + model3.predict(X_test))/3.
print(Yp_raw.shape[0])
Yp=np.sqrt(np.square(Yp_raw[:,:,:,0])+np.square(Yp_raw[:,:,:,1]))
#Yp=np.zeros((Yp_raw.shape[0],size,size))

#Yp = watershed(Yp_raw[:,:,:,1], (Yp_raw[:,:,:,0]>0.5))

#Yp = ndi.binary_fill_holes(Yp[j] - 1)

preds_test_upsampled = []
for i in range(len(Yp)):
    preds_test_upsampled.append(resize((Yp[i]),
                                       (x_test_original[i].shape[0],x_test_original[i].shape[1]),
                                       mode='constant', preserve_range=True))


for i in range(10):
    plt.imshow(Yp_raw[i])
    plt.savefig(folder+"photos/test_output"+str(i)+".png")
    plt.imshow(x_test[i,:,:,:])
    plt.savefig(folder+"photos/test_image"+str(i)+".png")
    plt.imshow(Yp[i])
    plt.savefig(folder+"photos/test_image"+str(i)+"mask2.png")


test_ids=np.unique(sample_submission.ImageId.values)
new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n], cutoff=0.5))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-2.csv', index=False)
