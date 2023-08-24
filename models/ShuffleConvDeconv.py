import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Activation, Conv2DTranspose, Conv2D, MaxPool2D, BatchNormalization, Lambda, DepthwiseConv2D, Concatenate
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class ShuffleConvDeconv():

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

def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x

def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret

def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x

def ShuffleNetV2(scale_factor=1.0,
                 shape=(224,224,3),
                 num_shuffle_units=[3,7,3],
                 bottleneck_ratio=1):
    
    out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}
  
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2**exp
    out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  #  calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    # create shufflenet architecture
    img_input = Input(shape=shape)
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = block(x, out_channels_in_stage,
                   repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   stage=stage + 2)

    if bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    shuffnet = Model(inputs=[img_input], outputs=[x])

    return shuffnet