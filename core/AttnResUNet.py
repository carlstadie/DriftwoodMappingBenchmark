from tensorflow.keras import layers, models, regularizers, backend as K


def repeat_elem(tensor, repeats):
    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': repeats})(tensor)


def residual_block(input_tensor, filter_size, filters, dropout=0.0, use_batchnorm=False):
    x = layers.Conv2D(filters, (filter_size, filter_size), padding='same')(input_tensor)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (filter_size, filter_size), padding='same')(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    shortcut = layers.Conv2D(filters, (1, 1), padding='same')(input_tensor)
    if use_batchnorm:
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def gating_signal(input_tensor, filters, use_batchnorm=False):
    x = layers.Conv2D(filters, (1, 1), padding='same')(input_tensor)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def attention_block(x, g, inter_channel):
    theta_x = layers.Conv2D(inter_channel, (2, 2), strides=(2, 2), padding='same')(x)
    phi_g = layers.Conv2D(inter_channel, (1, 1), padding='same')(g)

    upsample_g = layers.Conv2DTranspose(
        inter_channel, (3, 3),
        strides=(theta_x.shape[1] // g.shape[1], theta_x.shape[2] // g.shape[2]),
        padding='same')(phi_g)

    added = layers.add([theta_x, upsample_g])
    relu = layers.Activation('relu')(added)
    psi = layers.Conv2D(1, (1, 1), padding='same')(relu)
    sigmoid = layers.Activation('sigmoid')(psi)

    upsample_psi = layers.UpSampling2D(
        size=(x.shape[1] // sigmoid.shape[1], x.shape[2] // sigmoid.shape[2])
    )(sigmoid)
    upsample_psi = repeat_elem(upsample_psi, x.shape[3])

    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(x.shape[3], (1, 1), padding='same')(y)
    result = layers.BatchNormalization()(result)
    return result


def encoder_block(x, filter_size, filters, dropout, use_batchnorm):
    conv = residual_block(x, filter_size, filters, dropout, use_batchnorm)
    pool = layers.MaxPooling2D((2, 2))(conv)
    return conv, pool


def decoder_block(input_tensor, skip_tensor, filter_size, filters, dropout, use_batchnorm):
    gating = gating_signal(input_tensor, filters, use_batchnorm)
    attention = attention_block(skip_tensor, gating, filters)

    upsample = layers.UpSampling2D((2, 2))(input_tensor)
    concat = layers.Concatenate(axis=3)([upsample, attention])

    conv = residual_block(concat, filter_size, filters, dropout, use_batchnorm)
    return conv


def AttentionResUNet(input_shape, num_classes, dropout_rate=0.0, use_batchnorm=True):
    filter_size = 3
    base_filters = 64

    inputs = layers.Input(shape=input_shape)

    conv1, pool1 = encoder_block(inputs, filter_size, base_filters, dropout_rate, use_batchnorm)
    conv2, pool2 = encoder_block(pool1, filter_size, base_filters * 2, dropout_rate, use_batchnorm)
    conv3, pool3 = encoder_block(pool2, filter_size, base_filters * 4, dropout_rate, use_batchnorm)
    conv4, pool4 = encoder_block(pool3, filter_size, base_filters * 8, dropout_rate, use_batchnorm)

    center = residual_block(pool4, filter_size, base_filters * 16, dropout_rate, use_batchnorm)

    dec6 = decoder_block(center, conv4, filter_size, base_filters * 8, dropout_rate, use_batchnorm)
    dec7 = decoder_block(dec6, conv3, filter_size, base_filters * 4, dropout_rate, use_batchnorm)
    dec8 = decoder_block(dec7, conv2, filter_size, base_filters * 2, dropout_rate, use_batchnorm)
    dec9 = decoder_block(dec8, conv1, filter_size, base_filters, dropout_rate, use_batchnorm)

    output = layers.Conv2D(num_classes, (1, 1), padding='same')(dec9)
    output = layers.BatchNormalization()(output)
    output = layers.Activation('sigmoid')(output)

    model = models.Model(inputs=inputs, outputs=output, name='AttentionResUNet')
    return model
