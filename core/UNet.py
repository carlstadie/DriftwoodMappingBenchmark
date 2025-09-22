# core/UNet.py
from typing import Iterable, Union

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import regularizers as Kreg


def UNet(
    input_shape,
    input_label_channels: Union[int, Iterable[int]],
    dilation_rate: int = 1,
    layer_count: int = 64,
    l2_weight: float = 1e-4,
    dropout: float = 0.0,
    weight_file: str = None,
    summary: bool = False,
):
    """
    UNet with attention skip connections (robust version).

    Args:
        input_shape: (batch, height, width, channels)
        input_label_channels: int or iterable; number of output channels/classes.
        dilation_rate: dilation used inside conv blocks (encoder).
        layer_count: base number of filters.
        l2_weight: L2 regularization for conv layers (0 disables).
        dropout: dropout rate after BN in blocks (0 disables).
        weight_file: optional path to load weights.
        summary: print model summary if True.
    """
    # --- normalize num_classes
    if isinstance(input_label_channels, Iterable) and not isinstance(input_label_channels, (str, bytes)):
        try:
            num_classes = len(list(input_label_channels))
        except Exception:
            num_classes = int(input_label_channels)  # fallback
    else:
        num_classes = int(input_label_channels)

    kr = Kreg.l2(l2_weight) if l2_weight and l2_weight > 0 else None

    inp = layers.Input(shape=input_shape[1:], name="input")

    def conv_block(x, filters, dil=1):
        x = layers.Conv2D(filters, 3, padding="same", dilation_rate=dil, use_bias=True,
                          activation="relu", kernel_regularizer=kr)(x)
        x = layers.Conv2D(filters, 3, padding="same", dilation_rate=dil, use_bias=True,
                          activation="relu", kernel_regularizer=kr)(x)
        x = layers.BatchNormalization()(x)
        if dropout and dropout > 0:
            x = layers.Dropout(dropout)(x)
        return x

    # -------- encoder
    c1 = conv_block(inp, 1 * layer_count, dil=dilation_rate); p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 2 * layer_count, dil=dilation_rate);  p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 4 * layer_count, dil=dilation_rate);  p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3, 8 * layer_count, dil=dilation_rate);  p4 = layers.MaxPooling2D(2)(c4)
    c5 = conv_block(p4, 16 * layer_count, dil=dilation_rate)

    # -------- decoder with attention skips
    u6 = attention_up_and_concat(c5, c4); c6 = conv_block(u6, 8 * layer_count, dil=1)
    u7 = attention_up_and_concat(c6, c3); c7 = conv_block(u7, 4 * layer_count, dil=1)
    u8 = attention_up_and_concat(c7, c2); c8 = conv_block(u8, 2 * layer_count, dil=1)
    u9 = attention_up_and_concat(c8, c1); c9 = conv_block(u9, 1 * layer_count, dil=1)

    # -------- head
    out = layers.Conv2D(num_classes, 1, activation="sigmoid", kernel_regularizer=kr, name="mask")(c9)

    model = models.Model(inputs=inp, outputs=out, name="UNetAttention")

    if weight_file:
        model.load_weights(weight_file)
    if summary:
        model.summary()

    return model


def attention_up_and_concat(down_layer, skip_layer):
    """Upsample `down_layer`, attend `skip_layer`, then concat."""
    # upsample by 2 (robust to odd shapes via bilinear resize to match)
    up = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(down_layer)

    # runtime spatial align if needed
    def _resize_to(x, ref):
        # resize x to ref's HxW
        return tf.image.resize(x, size=tf.shape(ref)[1:3], method="bilinear")

    up_aligned = layers.Lambda(lambda t: _resize_to(t[0], t[1]))([up, skip_layer])

    # attention gate
    att = attention_block_2d(x=skip_layer, g=up_aligned)

    # concat along channels
    return layers.Concatenate(axis=-1)([up_aligned, att])


def attention_block_2d(x, g):
    """
    Standard attention gate:
      theta_x = W_x * x
      phi_g   = W_g * g (gate)
      f       = ReLU(theta_x + phi_g)
      rate    = sigmoid(W_f * f)
      att_x   = x * rate
    Ensures shapes match; projects to inter_channel >= 1.
    """
    ch_g = tf.shape(g)[-1]
    # try to derive a static channel count for inter_channel; fallback to runtime if unknown
    static_ch = g.shape[-1] if g.shape[-1] is not None else None
    inter = max(1, (static_ch // 4) if static_ch is not None else 16)  # safe default 16 if unknown

    theta_x = layers.Conv2D(inter, 1, padding="same", use_bias=True)(x)
    phi_g   = layers.Conv2D(inter, 1, padding="same", use_bias=True)(g)

    # Make sure spatial dims match in case of any rounding during pooling/upsampling
    def _resize_like(a, b):
        return tf.image.resize(a, size=tf.shape(b)[1:3], method="bilinear")

    phi_g = layers.Lambda(lambda t: _resize_like(t[0], t[1]))([phi_g, theta_x])

    f     = layers.Activation("relu")(layers.Add()([theta_x, phi_g]))
    psi   = layers.Conv2D(1, 1, padding="same", use_bias=True)(f)
    rate  = layers.Activation("sigmoid")(psi)

    # broadcast rate (H,W,1) over channel axis of x
    att_x = layers.Multiply()([x, rate])
    return att_x
