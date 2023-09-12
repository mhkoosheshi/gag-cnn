import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Add, Conv2DTranspose, Conv2D, MaxPool2D, BatchNormalization, Dropout, SeparableConv2D, Activation, UpSampling2D
from tensorflow.keras.models import Model

def Conv(input, filters, pool=False, activation="relu", bias=True, init='glorot_uniform', bn=True):
    x = input
    x = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    if bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if pool:
        x = MaxPool2D(pool_size=(2,2))(x)
    return x

def SepConv(input, filters, pool=False, activation="relu", bias=True, init='glorot_uniform', bn=True):
    x = input
    x = SeparableConv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    if bn:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if pool:
        x = MaxPool2D(pool_size=(2,2))(x)
    return x

def ResConv(input, filters, activation="relu", bias=True, init='glorot_uniform', bn=True):
    x = Conv(input, filters, pool=False, activation=activation, bias=bias, init=init, bn=bn)
    x = Conv(x, filters, pool=False, activation=activation, bias=bias, init=init, bn=bn)
    x_skip = input
    x = Add()([x, x_skip])
    # x = Activation(activation)(x)
    return x