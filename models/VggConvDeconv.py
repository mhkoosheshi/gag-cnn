import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPool2D, BatchNormalization, Dropout, SeparableConv2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from keras.layers import Lambda
from tensorflow.keras.models import Model

class VggConvDeconv():
    
    def __init__(self, backbone='vgg16', shape=(256,256)):
        self.shape = shape
        self.backbone = backbone
    
    def get_model(self):
        # obj input
        input = layers.Input(shape=self.shape)

        backbone = VggBackBone(backbone=self.backbone, shape=self.shape)
        encoder2 = VggHead(backbone.output, backbone='resnet18')
        out = encoderdecoder(input, encoder2)

        model = Model(inputs=[input, backbone.input], outputs=[out])
        
        return model



def VggHead(resoutput, backbone='resnet18'):
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(resoutput)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    return x

def encoderdecoder(input, features):
    
    x = Conv2D(8, (3, 3), activation='relu', strides=1, padding='same')(input)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(16, (3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(32, (3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(64, (3, 3), activation='relu', strides=1, padding='same')(x)

    x = layers.concatenatecnct = layers.concatenate([x, features])
    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(4, (1,1), activation='linear', padding='same')(x)

    return x

def VggBackBone(backbone='vgg16', shape=(512,512,3)):
    
    
    if backbone=='vgg16':
        model = VGG16(weights='imagenet', include_top=False, input_shape=shape)
        return model

    elif backbone=='vgg19':
        model = VGG19(weights='imagenet', include_top=False, input_shape=shape)
        return model

    elif backbone=='vgg16s2d':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=shape)
        base_model.trainable = True
        n_freeze = 0
        for layer in base_model.layers[1:n_freeze+1]:
            layer.trainable = False

        x0 = tf.keras.Input(shape=(224,224,3))
        x1 = base_model.layers[1](x0)
        x1 = base_model.layers[2](x1)
        s2d1 = Lambda(lambda x:tf.nn.space_to_depth(x,2), name='s2d1')
        x2 = s2d1(x1)
        x2 = Conv2D(64, (3,3), padding='same', activation='relu')(x2)

        x2 = base_model.layers[4](x2)
        x2 = base_model.layers[5](x2)
        s2d2 = Lambda(lambda x:tf.nn.space_to_depth(x,2), name='s2d2')
        x3 = s2d2(x2)
        x3 = Conv2D(128, (3,3), padding='same', activation='relu')(x3)

        x3 = base_model.layers[7](x3)
        x3 = base_model.layers[8](x3)
        x3 = base_model.layers[9](x3)
        s2d3 = Lambda(lambda x:tf.nn.space_to_depth(x,2), name='s2d3')
        x4 = s2d3(x3)
        x4 = Conv2D(256, (3,3), padding='same', activation='relu')(x4)

        x4 = base_model.layers[11](x4)
        x4 = base_model.layers[12](x4)
        x4 = base_model.layers[13](x4)
        s2d4 = Lambda(lambda x:tf.nn.space_to_depth(x,2), name='s2d4')
        x5 = s2d4(x4)
        x5 = Conv2D(512, (3,3), padding='same', activation='relu')(x5)

        x5 = base_model.layers[15](x5)
        x5 = base_model.layers[16](x5)
        x5 = base_model.layers[17](x5)
        s2d5 = Lambda(lambda x:tf.nn.space_to_depth(x,2), name='s2d5')
        x6 = s2d5(x5)
        x6 = Conv2D(512, (3,3), padding='same', activation='relu')(x6)
        model = Model(inputs=[x0], outputs=[x6])
        return model