import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Activation, Input, Conv2DTranspose
from tensorflow.keras.constraints import MaxNorm
from keras.layers import Lambda
from tensorflow.keras.models import Model

class BaseConvDeconv():
    '''
    Works on stack True
    '''
    
    def __init__(self, shape=(512,512,3), activation='relu'):
        self.shape=shape
        self.activation = activation
    
    def get_model(self):
        input1 = Input(shape=self.shape)
        input2 = Input(shape=self.shape)
        encoder1 = encoder(input1, name='conv1', activation=self.activation)
        encoder2 = encoder(input2, name='conv2', activation=self.activation)
        
        cnct = layers.concatenate([encoder1, encoder2])
        
        up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', name='tconv1_1')(cnct)
        conv1 = Conv2D(256, (3, 3), activation=self.activation, padding='same', name='upconv1_1')(up1)
        conv1 = Conv2D(256, (3, 3), activation=self.activation, padding='same', name='upconv1_2')(conv1)

        up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='tconv2_1')(conv1)
        conv2 = Conv2D(128, (3, 3), activation=self.activation, padding='same', name='upconv2_1')(up2)
        conv2 = Conv2D(128, (3, 3), activation=self.activation, padding='same', name='upconv2_2')(conv2)

        up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='tconv3_1')(conv2)
        conv3 = Conv2D(64, (3, 3), activation=self.activation, padding='same', name='upconv3_1')(up3)
        conv3 = Conv2D(64, (3, 3), activation=self.activation, padding='same', name='upconv3_2')(conv3)

        up4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='tconv4_1')(conv3)
        conv4 = Conv2D(32, (3, 3), activation=self.activation, padding='same', name='upconv4_1')(up4)
        conv4 = Conv2D(32, (3, 3), activation=self.activation, padding='same', name='upconv4_2')(conv4)
        conv6 = Conv2D(4, (1, 1), activation='linear', padding='same', name='upconv5_3')(conv4)
        

        based_model = Model(inputs=[input1, input2], outputs=[conv6])

        return based_model

def Convlayer(input, outdim, is_batchnorm, name, activation='relu'):
    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same", name=name+'_1')(input)
    if is_batchnorm:
        x =BatchNormalization(name=name + '_1_bn')(x)
    x = Activation(activation,name=name + '_1_act')(x)

    x = Conv2D(outdim, (3, 3), strides=(1, 1), padding="same", name=name+'_2')(x)
    if is_batchnorm:
        x = BatchNormalization(name=name + '_2_bn')(x)
    x = Activation(activation, name=name + '_2_act')(x)
    return x

def encoder(inputs, name, activation='relu'):
    conv1 = Convlayer(inputs, 32, is_batchnorm=True, name=name+'1', activation=activation)
    pool1 = MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = Convlayer(pool1, 64, is_batchnorm=True, name=name+'2', activation=activation)
    pool2 = MaxPool2D(pool_size=(2,2))(conv2)
    conv3 = Convlayer(pool2, 128, is_batchnorm=True, name=name+'3', activation=activation)
    pool3 = MaxPool2D(pool_size=(2,2))(conv3)
    conv4 = Convlayer(pool3, 256, is_batchnorm=True, name=name+'4', activation=activation)
    pool4 = MaxPool2D(pool_size=(2,2))(conv4)
    conv5 = Conv2D(512, (3, 3), activation=activation, padding='same', name=name+'5')(pool4)
    conv5 = Conv2D(512, (3, 3), activation=activation, padding='same', name=name+'6')(conv5)
    outputs = conv5

    return outputs



#####################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPool2D, BatchNormalization, Dropout, SeparableConv2D, Activation
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model

class GAG_CNN():

    def __init__(self, shape=(256,256,3)):
        self.shape = shape

    def get_model(self):
        # obj input
        input1 = layers.Input(shape=self.shape)
        # iso input
        input2 = layers.Input(shape=self.shape)

        backbone = Backbone(input2, shape=self.shape)
        encoder2 = Head(backbone)
        out = encoderdecoder(input1, encoder2)

        model = Model(inputs=[input1, input2], outputs=[out], name='GAG_CNN')

        return model



def Head(input):
    x = input
    # x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    return x

def encoderdecoder(input, features):

    x = Conv2D(8, (3, 3), strides=1, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(16, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(32, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    

    x = layers.concatenatecnct = layers.concatenate([x, features])
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)

    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(x)

    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(4, (1,1), activation='linear', padding='same')(x)

    return x

def Backbone(input, shape=(512,512,3)):
    x = Conv2D(8, (3, 3), strides=1, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(16, (3, 3), activation='relu', strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(32, (3, 3), activation='relu', strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    return x


#########################################################

class GAG_CNN():

    def __init__(self, shape=(256,256,3), init="glorot_uniform", bias=False, activation="relu"):
        self.shape = shape
        self.init = init
        self.bias = bias
        self.activation = activation

    def get_model(self):
        # obj input
        input1 = layers.Input(shape=self.shape)
        # iso input
        input2 = layers.Input(shape=self.shape)

        backbone = Backbone(input2, shape=self.shape, init=self.init, bias=self.bias, activation=self.activation)
        encoder2 = Head(backbone, init=self.init, activation=self.activation)
        out = encoderdecoder(input1, encoder2, init=self.init, activation=self.activation, bias=self.bias)

        model = Model(inputs=[input1, input2], outputs=[out], name='GAGCNN')

        return model



def Head(input, init="glorot_uniform", activation="relu"):
    x = input
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def encoderdecoder(x, features, init="glorot_uniform", bias=False, activation="relu"):

    x = Conv2D(16, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(32, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(32, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(128, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(128, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)


    x = layers.concatenatecnct = layers.concatenate([x, features])
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2DTranspose(256, (3, 3), strides=2, padding='same', kernel_initializer=init, use_bias=bias)(x)

    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same', kernel_initializer=init, use_bias=bias)(x)

    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same', kernel_initializer=init, use_bias=bias)(x)

    x = Conv2D(32, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same', kernel_initializer=init, use_bias=bias)(x)

    x = Conv2D(16, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = Conv2D(16, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(8, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(8, (3, 3), padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(4, (1,1), activation='sigmoid', padding='same',
              #  kernel_initializer=init,
               kernel_initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1, seed=42),
               use_bias=bias)(x)

    return x

def Backbone(x, shape=(512,512,3), init="glorot_uniform", bias=False, activation="relu"):

    x = Conv2D(16, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(32, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(32, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    # x = Conv2D(64, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    # x = BatchNormalization()(x)
    # x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(128, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(128, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    # x = Conv2D(128, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    # x = BatchNormalization()(x)
    # x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)

    x = Conv2D(256, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = Conv2D(256, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    # x = Conv2D(256, (3, 3), strides=1, padding='same', kernel_initializer=init, use_bias=bias)(x)
    # x = BatchNormalization()(x)
    # x = Activation(activation)(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    return x