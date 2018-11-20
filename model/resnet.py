from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, TimeDistributed
from keras import backend as K
from model.RoIpoolingConv import RoIpoolingConv
from model.BatchNormalization import BatchNormalization

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    nb_filter1,nb_filter2,nb_filter3=filters
    bn_axis=3
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'
    x=Convolution2D(nb_filter1,(1,1),name=conv_name_base+'2a',
                    trainable=trainable)(input_tensor)
    x=BatchNormalization(axis=bn_axis,name=bn_name_base+'2a')(x)
    x=Activation('relu')(x)

    x=Convolution2D(nb_filter2,(kernel_size,kernel_size),padding='same',
                    name=conv_name_base+'2b',trainable=trainable)(x)
    x=BatchNormalization(axis=bn_axis,name=bn_name_base+'2b')(x)
    x=Activation('relu')(x)

    x=Convolution2D(nb_filter3,(1,1),name=conv_name_base+'2c',
                    trainable=trainable)(x)
    x=BatchNormalization(axis=bn_axis,name=bn_name_base+'2c')(x)

    x=Add()[x,input_tensor]
    x=Activation('relu')(x)
    return x


def identity_block_td(input_tensor,kernel_size,filters,stage,block,trainable=True):
    nb_filter1,nb_filter2,nb_filter3=filters
    bn_axis=3
    conv_name_base='res'+str(stage)+block+'_branch'
    bn_name_base='bn'+str(stage)+block+'_branch'

    x=TimeDistributed(Convolution2D(nb_filter1,(1,1),trainable=trainable,
                                    kernel_initializer='normal'),name=conv_name_base+'2a')(input_tensor)
    x=TimeDistributed(BatchNormalization(axis=bn_axis),name=bn_name_base+'2b')(x)
    x=Activation('relu')(x)
    x = TimeDistributed(
        Convolution2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',
                      padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'),
                        name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x



def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                      trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(
        input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

    # conv block time distributed

    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = TimeDistributed(Convolution2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Convolution2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    shortcut = TimeDistributed(Convolution2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(input_tensor=None,trainable=False):
    input_shape=(None,None,3)
    if input_tensor is None:
        img_input=Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input=Input(tensor=input_tensor,shape=input_shape)

        else:
            img_input=input_tensor
    bn_axis=3

    x=ZeroPadding2D((3,3))(img_input)
    x=Convolution2D(64,(7,7),strides=(2,2),name='conv1',trainable=trainable)(x)












