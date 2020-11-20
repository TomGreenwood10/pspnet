"""
File containing class for biulding pyramid scene parsing network (PSPNet).
"""

from keras.layers import AveragePooling2D, Conv2D, BatchNormalization, \
    Activation, UpSampling2D, Input, Concatenate
from keras.models import Model

from BaseModel import SegmentationModel, get_pool_size
from ResNet.builder import res_block


class PSPNet(SegmentationModel):
    """
    Class for Pyramid Scene Parsing Network with ResNet feature map.
    """

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
            if i != 0 and i % self.pooling_freq_per_resblock == 0:
                pool_size = get_pool_size(x.shape)
                x = AveragePooling2D((pool_size, pool_size), padding='same')(x)
                filters *= pool_size
                filters = int(filters)
            x = Conv2D(filters, (1, 1))(x)  # for resizing nfilters
            x = res_block(x, filters, self.activations, f'resblock_{i + 1}',
                          self.block_types, self.n_convs, self.conv_in_skip)

            # Assign an aux output if there is room and it isn't the same at 
            # the main output. Up sample to match input shape
            if (i >= self.aux_output_block 
                and aux_output is False 
                and i != self.blocks - 1):
                x = Conv2D(1, (1, 1), name='conv_aux_output')(x)
                x = BatchNormalization(name='bn_aux_output')(x)
                upsample_scalar = int(self.input_shape[-2] / 
                                      x._keras_shape[-2])
                x = UpSampling2D((upsample_scalar, upsample_scalar), 
                    interpolation='bilinear', name='aux_upsample')(x)
                aux_output = Activation(self.output_activations, 
                    name='AUX_OUTPUT')(x)

        x = pyramid_module(x)
        upsample_scalar = int(self.input_shape[-2] / x._keras_shape[-2])
        # Conv with 5 filters for 5 classes (in practice set) - change as req.
        x = Conv2D(5, (1, 1), name='conv_main_output')(x)
        x = BatchNormalization(name='bn_main_output')(x)
        x = UpSampling2D((upsample_scalar, upsample_scalar),
            interpolation='bilinear', name='main_upsample')(x)
        main_output = Activation(self.output_activations,
            name='MAIN_OUTPUT')(x)
        outputs = [aux_output, main_output] if aux_output else main_output
        self.net = Model(inputs, outputs, name='ResNet')


def pyramid_branch(inputs, pooled_size, activation=None, 
                   batch_normalisation=False, n_branches=4):
    if inputs.shape[1] != inputs.shape[2]:
        raise ValueError("Inputs should be square but found to have shape " +
                         f"{inputs.shape}")

    pool_dim = int(inputs.shape[1]) / pooled_size
    pool_dim = int(pool_dim)
    pool_size = (pool_dim, pool_dim)
    name = f'{pooled_size}x{pooled_size}_pyramid_branch'

    x = AveragePooling2D(pool_size=pool_size, padding='same',
        name='pool_' + name)(inputs)
    x = Conv2D(int(int(x.shape[-1]) / n_branches), (1, 1), padding='same', 
        name='conv_' + name)(x)
    if batch_normalisation:
        x = BatchNormalization(name='bn_' + name)(x)
    if activation:
        x = Activation(activation, name=activation + '_' + name)(x)
    x = UpSampling2D((pool_dim, pool_dim), interpolation='bilinear',
        name='upsamp_' + name)(x)
    return x


def pyramid_module(inputs, activation=None, batch_normalisation=False, 
                   pool_sizes=(1, 2, 5, 10)):
    pyramid_outputs = []
    for pool_size in pool_sizes:
        pyramid_outputs.append(pyramid_branch(inputs,
                                              pool_size,
                                              activation,
                                              batch_normalisation,
                                              n_branches=len(pool_sizes)))
    x = Concatenate(axis=3)(pyramid_outputs + [inputs])
    return x
