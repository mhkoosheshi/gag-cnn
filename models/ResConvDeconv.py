import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose

class ResConvDeconv():
    
    def __init__(self, shape=(512,512,3), backbone='resnet18'):
        self.shape = shape
        self.backbone = backbone

    def get_model(self):
        # obj input
        input = layers.Input(shape=self.shape)

        backbone = ResBackBone(backbone=self.backbone, shape=self.shape)
        encoder2 = ResHead(backbone.output, backbone='resnet18')
        out = encoderdecoder(input, encoder2)

        model = Model(inputs=[input, backbone.input], outputs=[out])
        
        return model

def ResBackBone(backbone='resnet18', shape=(512,512,3)):
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=shape)
    
    if backbone=='resnet50':
        model = base_model
        return model

    elif backbone=='resnet18':
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv3_block2_out').output)
        return model

    elif backbone=='resnet14':
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv3_block1_out').output)
        return model

    elif backbone=='resnet20':
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv3_block3_out').output)
        return model

    elif backbone=='resnet26':
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block1_out').output)
        return model

    elif backbone=='resnet34':
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block2_out').output)
        return model

def ResHead(resoutput, backbone='resnet18'):
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
    x = Conv2DTranspose(32, (3, 3), strides=1, padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(16, (3, 3), strides=1, padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(8, (3, 3), strides=1, padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    x = Conv2D(4, (1,1), activation='relu', padding='same')(x)

    return x


