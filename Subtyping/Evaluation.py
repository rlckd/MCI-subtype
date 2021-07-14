from __future__ import absolute_import, division, print_function, unicode_literals
import os
import glob
import math
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.layers import Conv3D, ZeroPadding3D, MaxPooling3D
from tensorflow.keras.layers import Input, Flatten, add, Dense, Activation, Reshape
from tensorflow.keras.layers import Dropout, GlobalAveragePooling3D, concatenate, Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import scipy.ndimage
import nibabel as nib



model = load_model('DenseNet_model_whole.h5')






ff = sorted(glob.glob("data/*.nii.gz"))


for f in range(len(ff)):
    a = nib.load(ff[f])
    data = a.get_data()

    brain = data

    brain = np.asarray(brain, dtype='float32')

    brain = brain.reshape(1,91,109,91,1)


    y_pred = model.predict(brain)
    print(ff[f])
    print(y_pred)

