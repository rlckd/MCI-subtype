from __future__ import absolute_import, division, print_function, unicode_literals
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.layers import Conv3D, ZeroPadding3D, MaxPooling3D, PReLU, LeakyReLU
from tensorflow.keras.layers import Input, Flatten, add, Dense, Activation, Reshape
from tensorflow.keras.layers import Dropout, GlobalAveragePooling3D, concatenate, Softmax
from tensorflow.keras.layers import BatchNormalization, Concatenate, AveragePooling3D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.client import device_lib
from tensorflow.keras.utils import multi_gpu_model, to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
#from tensorflow.config.gpu import set_per_process_memory_growth
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import backend as K

#tf.debugging.set_log_device_placement(True)

#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(len(gpus), "Physical GPUs")

tf.config.experimental.set_memory_growth(gpu,True)


##Focal loss for imbalanced data

def binary_focal_loss(gamma=2., alpha=.25):

    def binary_focal_loss_fixed(y_true, y_pred):

        pt_1 = tf.where(tf.equal(y_true,1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true,0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()

        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return binary_focal_loss_fixed



if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

def decayed_learning_rate(step):
    return initial_learning_rate * decay_rate ^ (step / decay_steps)


def exp_decay(epoch):
    initial_lrate = 0.0001
    k = 0.04
    lrate = initial_lrate * math.exp(-k)
    return lrate

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 2.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

class DataFeedGenerator(tf.keras.utils.Sequence):

    def __init__(self,list_IDs,x,y,batch_size=32, dim=(91,109,91), n_channels=1, n_classes=1, shuffle=False, name="Training"):
        self.dim = dim
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.Y = y
        self.X = x
        self.currentX = None
        self.currentY = None
        self.batch_index = 0
        self.n_channels = n_channels
        self.classes = n_classes
        self.shuffle = shuffle
        self.name = name
        self.on_epoch_end()

    def __len__(self):
        n = math.ceil(self.X.shape[0] / self.batch_size)
        print(self.name, "__len__", n)
        return n

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X,Y = self.__data_generation(list_IDs_temp)
        
        return X, Y


    def __data_generation(self,list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.X[ID]
            Y[i,] = self.Y[ID]
        return X, Y



## need training data (hdf5 format) 

x_train = HDF5Matrix('ADNI_GM_data_kfold1.hdf5', 'brain_train')
y_train = HDF5Matrix('ADNI_GM_data_kfold1.hdf5', 'y_train')

x_test = HDF5Matrix('ADNI_GM_data_kfold1.hdf5', 'brain_test')
y_test = HDF5Matrix('ADNI_GM_data_kfold1.hdf5', 'y_test')


def dense_block(x, blocks, name):

    for i in range(blocks):
        x = conv_block(x,32,name=name + '_block' + str(i + 1))

    return x

def transition_block(x, reduction, name):

    bn_axis = 4
    weight_decay = 1e-4
    eps = 1.1e-5
    x = BatchNormalization(axis=4, momentum=0.8, epsilon=eps, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), name=name + '_bn')(x)
    x = LeakyReLU(alpha=0.3, name=name + '_relu')(x)
    x = Conv3D(int(x.shape[bn_axis] * reduction), kernel_size=(1,1,1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay), name=name + '_conv')(x)
    x = AveragePooling3D((2,2,2), strides=(2,2,2), name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name):

    bn_axis = 4
    weight_decay = 1e-4
    eps = 1.1e-5
    x1 = BatchNormalization(axis=4, momentum=0.8, epsilon=eps, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), name=name + '_0_bn')(x)
    x1 = LeakyReLU(alpha=0.3, name=name + '_0_relu')(x1)
    x1 = Conv3D(4 * growth_rate, kernel_size=(1,1,1), padding='same', use_bias=False, name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=4, momentum=0.8, epsilon=eps, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), name=name + '_1_bn')(x1)
    x1 = LeakyReLU(alpha=0.3, name=name + '_1_relu')(x1)
    x1 = Conv3D(growth_rate, kernel_size=(5,5,5), padding='same', use_bias=False, name=name + '_2_conv')(x1)
    x1 = Dropout(0.2, name=name + '_drop')(x1)
    x = Concatenate(axis=bn_axis, name = name + '_concat')([x, x1])
    return x 

def DenseNet(x, blocks, name):
    
    weight_decay = 1e-4
    eps = 1.1e-5
    #initial_layer
    x = Conv3D(16, kernel_size=(5,5,5), padding='same', use_bias=False, name=name + 'conv1/conv')(x)
    x = BatchNormalization(axis=4, momentum=0.8, epsilon=eps, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay), name=name + 'conv1/bn')(x)
    x = LeakyReLU(alpha=0.3, name=name + 'conv1/relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), strides=2, name=name + 'pool1')(x)

    x = dense_block(x, blocks[0], name=name + 'conv2')
    x = transition_block(x, 0.5, name=name + 'pool2')
    x = dense_block(x, blocks[1], name=name + 'conv3')
    x = transition_block(x, 0.5, name=name + 'pool3')
    x = dense_block(x, blocks[2], name=name + 'conv4')
    x = transition_block(x, 0.5, name=name + 'pool4')
    x = dense_block(x, blocks[3], name=name + 'conv5')

    x = GlobalAveragePooling3D(name=name + 'avg_pool')(x)
    return x

epochs = 200
batch_size = 16
learningRate = 0.0001

#sgd = SGD(lr=learningRate, decay=decay, momentum=0.9, nesterov=True)
#sgd = SGD(lr=learningRate, momentum=0.9)
#adam = Adam(lr = learningRate, beta_1=0.9, beta_2=0.999, amsgrad=False)
#lrate = LearningRateScheduler(exp_decay)

initial_learning_rate = 0.0001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = initial_learning_rate,
        decay_steps = 10000,
        decay_rate=0.90,
        staircase=True)
adam = Adam(learning_rate = lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False)
#sgd = SGD(lr=lr_schedule, momentum=0.9)
trainingGen = DataFeedGenerator(list_IDs=list(range(0,x_train.shape[0])), x=x_train, y=y_train, batch_size = batch_size, name="Training Gen")
validationGen = DataFeedGenerator(list_IDs=list(range(0,x_test.shape[0])), x=x_test, y=y_test, batch_size = batch_size, name="Validation Gen")

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))


with strategy.scope():


    whole_brain = keras.Input(shape=(91,109,91,1), name='Input1')

    _X = DenseNet(whole_brain, [2,2,2,2], name='brain_')

    x = Dense(124, kernel_regularizer=l2(l=0.01))(_X)
#    x = BatchNormalization(momentum=0.9, epsilon=1.1e-5, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)
    x = Activation('relu')(x)
#    x = Dense(10, activation='relu', kernel_regularizer=l2(l=0.001))(x)
#    x = BatchNormalization(momentum=0.9, epsilon=1.1e-5, gamma_regularizer=l2(1e-4), beta_regularizer=l2(1e-4))(x)


    output = Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=whole_brain, outputs=output)


    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
#    model.compile(loss=[binary_focal_loss(gamma=2., alpha=.3)], optimizer=adam, metrics=['accuracy'])
    checkpoint = ModelCheckpoint("weights-{epoch:02d}-{val_accuracy:.4f}.hdf5", monitor='val_accuracy', verbose=1, save_freq='epoch' ,save_best_only=False, mode='max')


model.summary()
history = model.fit_generator(generator = trainingGen, verbose=1, validation_data=validationGen, callbacks=[checkpoint], epochs = epochs, shuffle=False)

print(history.history.keys())


model.save('DenseNet_model_whole.h5')




