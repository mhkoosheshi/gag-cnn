import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, BatchNormalization, Dropout, Activation, Input, concatenate, Conv2DTranspose
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.constraints import MaxNorm
from keras.layers import Lambda
from tensorflow.keras.models import Model

class BaseConvDeconv():
    def __init__(self, shape):
        self.shape=shape
    
    def get_model(self):
        input1 = Input(shape=self.shape)
        input2 = Input(shape=self.shape)
        encoder1 = encoder(input1)
        encoder2 = encoder(input2)
        
        cnct = layers.concatenate([encoder1, encoder2])
        
        up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', name='tconv1_1')(cnct)
        conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='upconv1_1')(up1)
        conv1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='upconv1_2')(conv1)

        up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='tconv2_1')(conv1)
        conv2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='upconv2_1')(up2)
        conv2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='upconv2_2')(conv2)

        up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='tconv3_1')(conv2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='upconv3_1')(up3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='upconv3_2')(conv3)

        up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='tconv4_1')(conv3)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='upconv4_1')(up4)
        conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='upconv4_2')(conv4)

        up5 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='tconv5_1')(conv4)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', name='upconv5_1')(up5)
        conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', name='upconv5_2')(conv5)
        conv6 = Conv2D(4, (1, 1), activation='linear', padding='same', name='upconv5_3')(conv5)

        based_model = Model(inputs=[input1, input2], outputs=[conv6[:,:,:,0], conv6[:,:,:,1], conv6[:,:,:,2], conv6[:,:,:,3]])

        return based_model

def Convlayer(input, outdim, is_batchnorm, name):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same", name=name+'_1')(input)
    if is_batchnorm:
        x =BatchNormalization(name=name + '_1_bn')(x)
    x = Activation('relu',name=name + '_1_act')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same", name=name+'_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation('relu', name=name + '_2_act')(x)
    return x

def encoder(inputs):
    conv1 = Convlayer(inputs, 32, is_batchnorm=True, name='conv1')
    pool1 = MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = Convlayer(pool1, 64, is_batchnorm=True, name='conv2')
    pool2 = MaxPool2D(pool_size=(2,2))(conv2)
    conv3 = Convlayer(pool2, 128, is_batchnorm=True, name='conv3')
    pool3 = MaxPool2D(pool_size=(2,2))(conv3)
    conv4 = Convlayer(pool3, 256, is_batchnorm=True, name='conv4')
    pool4 = MaxPool2D(pool_size=(2,2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    outputs = conv5

    return outputs