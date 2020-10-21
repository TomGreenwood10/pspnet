# Functions to build a ResNet framework

from keras.layers import (Conv2D, BatchNormalization, Activation, Add, Input, 
    Dense, AveragePooling2D, UpSampling2D)
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from BaseModel import SegmentationModel, get_pool_size


class ResNet(SegmentationModel):

    def __init__(self,
                 blocks, 
                 start_filters,
                 aux_output_depth=0.65,
                 activation='relu',
                 output_activations='softmax',
                 n_convs=3,
                 pooled=True,
                 pooling_freq_per_resblock=2,
                 block_types='bottleneck',
                 conv_in_skip=False,
                 **kwargs):

        super().__init__()
        self.set_attr(kwargs)
        self.blocks = blocks
        self.start_filters = start_filters
        self.aux_output_block = blocks * aux_output_depth
        self.activations = activation
        self.output_activations = output_activations
        self.n_convs = n_convs
        self.pooled = pooled
        self.pooling_freq_per_resblock = pooling_freq_per_resblock
        self.block_types = block_types
        self.conv_in_skip = conv_in_skip
        self.build()

    def build(self):
        inputs = Input(self.input_shape)
        filters = self.start_filters
        aux_output = False
        x = inputs
        for i in range(self.blocks):
            pool_size = 1
            if i != 0 and i % self.pooling_freq_per_resblock == 0:
                pool_size = get_pool_size(x.shape)
                x = AveragePooling2D((pool_size, pool_size), padding='same')(x)
                filters *= pool_size
                filters = int(filters)
            x = Conv2D(filters, (1, 1))(x)  # for resizing nfilters
            x = res_block(x, filters, self.activations, f'resblock_{i+1}', 
                          self.block_types, self.n_convs, self.conv_in_skip)
            
            # Assign an aux output if there is room and it isn't the same at 
            #  the main output. Up sample to match input shape
            if i >= self.aux_output_block and aux_output is False and i != self.blocks-1:
                x = Conv2D(1, (1, 1), name='conv_aux_output')(x)
                x = BatchNormalization(name='bn_aux_output')(x)
                upsample_scalar = int(self.input_shape[-2] / x._keras_shape[-2])
                x = UpSampling2D((upsample_scalar, upsample_scalar), interpolation='bilinear', name='aux_upsample')(x)
                aux_output = Activation(self.output_activations, name='AUX_OUTPUT')(x)

        upsample_scalar = int(self.input_shape[-2] / x._keras_shape[-2])
        x = Conv2D(1, (1, 1), name='conv_main_output')(x)
        x = BatchNormalization(name='bn_main_output')(x)
        x = UpSampling2D((upsample_scalar, upsample_scalar), interpolation='bilinear', name='main_upsample')(x)
        main_output = Activation(self.output_activations, name='MAIN_OUTPUT')(x)
        outputs = [aux_output, main_output] if aux_output is not False else main_output
        self.net = Model(inputs, outputs, name='ResNet')


def res_block_inner(inputs, n_convs, filters, activation, block_name):
    if not type(n_convs) is int:
        raise TypeError('n_convs must be an integer >= 1')
    if n_convs < 1:
        raise ValueError('n_convs must be >= 1')
    names = [f'{block_name}_3x3_conv', f'{block_name}_bn', f'{block_name}_{activation}']
    x = inputs
    conv_counter = 1
    while conv_counter < n_convs:
        x = Conv2D(filters, (3, 3), padding='same', name=names[0] + str(conv_counter))(x)
        x = BatchNormalization(name=names[1] + str(conv_counter))(x)
        x = Activation(activation, name=names[2] + str(conv_counter))(x)
        conv_counter += 1
    x = Conv2D(filters, (3, 3), padding='same', name=names[0] + str(conv_counter))(x)
    x = BatchNormalization(name=names[1] + str(conv_counter))(x)
    return x


def res_block_inner_bottleneck(inputs, filters, activation, block_name):
    names = [f'{block_name}_1x1_conv_start', f'{block_name}_bn1', f'{block_name}_{activation}1',
            f'{block_name}_3x3_conv', f'{block_name}_bn2', f'{block_name}_{activation}2',
            f'{block_name}_1x1_conv_end', f'{block_name}_bn3']
    x = Conv2D(filters, (1, 1), name=names[0])(inputs)
    x = BatchNormalization(name=names[1])(x)
    x = Activation(activation, name=names[2])(x)
    x = Conv2D(filters, (3, 3), padding='same', name=names[3])(x)
    x = BatchNormalization(name=names[4])(x)
    x = Activation(activation, name=names[5])(x)
    x = Conv2D(filters, (1, 1), name=names[6])(x)
    x = BatchNormalization(name=names[7])(x)
    return x


def skip_connection(inputs, con_in_skip=False, filters=None, block_name=None):
    names = [f'{block_name}_3x3_skip_conv', f'{block_name}_skip_bn']
    if con_in_skip:
        x = Conv2D(filters, (3, 3), padding='same', name=names[0])(inputs)
        x = BatchNormalization(name=names[1])(x)
    else:
        x = inputs
    return x


def res_block(inputs, filters, activation, block_name,
              inner_type='bottleneck', n_convs=3, con_in_skip=False):
    if inner_type == 'bottleneck':
        inner = res_block_inner_bottleneck(inputs, filters, activation, block_name)
    else:
        inner = res_block_inner(inputs, n_convs, filters, activation, block_name)
    skip = skip_connection(inputs, con_in_skip, filters, block_name)
    x = Add(name=f'{block_name}_add')([inner, skip])
    x = Activation(activation, name=f'{block_name}_final_{activation}')(x)
    return x

