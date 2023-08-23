import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPool2D, BatchNormalization, Dropout, SeparableConv2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model

class MobileConvDeconv():

    def __init__(self, backbone='mobilenet', shape=(256,256,3)):
        self.shape = shape
        self.backbone = backbone

    def get_model(self):
        # obj input
        input = layers.Input(shape=self.shape)

        backbone = MobileBackBone(backbone=self.backbone, shape=self.shape)
        encoder2 = MobileHead(backbone.output, backbone=self.backbone)
        out = encoderdecoder(input, encoder2)

        model = Model(inputs=[input, backbone.input], outputs=[out], name='MobileConvDeconv_'+ self.backbone)

        return model



def MobileHead(resoutput, backbone='mobilenet'):
    x = resoutput
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
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
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(64, (3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Conv2D(64, (3, 3), activation='relu', strides=1, padding='same')(x)

    x = layers.concatenatecnct = layers.concatenate([x, features])
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(16, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(8, (3, 3), strides=2, padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(4, (1,1), activation='linear', padding='same')(x)

    return x

def MobileBackBone(backbone='mobilenet', shape=(512,512,3)):
        if backbone == 'mobilenetv2':
          backbone = tf.keras.applications.MobileNetV2(input_shape=shape, alpha=1, weights="imagenet", 
                                        include_top=False)
        elif backbone == 'mobilenet':
          backbone = tf.keras.applications.MobileNet(input_shape=shape, alpha=1, weights="imagenet", 
                                        include_top=False,
                                        depth_multiplier=1, dropout=0.001)
        
        return backbone