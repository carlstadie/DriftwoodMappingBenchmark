import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

# --- Keep your window_partition and window_reverse exactly as is ---
def window_partition(x, window_size):
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, [B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size[0], window_size[1], C])
    return windows

def window_reverse(windows, window_size, H, W):
    window_h, window_w = window_size
    H = tf.cast(H, tf.int32)
    W = tf.cast(W, tf.int32)
    B = tf.shape(windows)[0] // (tf.math.floordiv(H, window_h) * tf.math.floordiv(W, window_w))
    x = tf.reshape(windows, [
        B,
        tf.math.floordiv(H, window_h),
        tf.math.floordiv(W, window_w),
        window_h,
        window_w,
        -1
    ])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, -1])
    return x

class DropPath(layers.Layer):
    def __init__(self, drop_prob=0., **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if (not training) or self.drop_prob == 0.:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        binary_tensor = tf.floor(random_tensor)
        return (x / keep_prob) * binary_tensor

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads=4,
                 attn_drop_rate=0.1, proj_drop_rate=0.1):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.scale       = self.head_dim ** -0.5

        self.qkv  = layers.Dense(dim * 3, use_bias=True)
        self.proj = layers.Dense(dim)
        self.attn_dropout = layers.Dropout(attn_drop_rate)
        self.proj_dropout = layers.Dropout(proj_drop_rate)

        Wh, Ww = window_size
        num_rel_positions = (2*Wh - 1) * (2*Ww - 1)
        self.relative_position_bias_table = self.add_weight(
            shape=(num_rel_positions, num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table"
        )
        coords_h = tf.range(Wh)
        coords_w = tf.range(Ww)
        coords   = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords   = tf.reshape(coords, [2, -1])
        rel_coords = coords[:, :, None] - coords[:, None, :]
        rel_coords = tf.transpose(rel_coords, [1,2,0])
        rel_coords = rel_coords + tf.constant([Wh-1, Ww-1])
        rel_index  = rel_coords[...,0]*(2*Ww-1) + rel_coords[...,1]
        self.relative_position_index = tf.cast(rel_index, tf.int32)

    def call(self, x, mask=None, training=None):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, C//self.num_heads))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        #k = tf.nn.l2_normalize(k, axis=-1)  # Helps stabilize dot product

        attn = tf.matmul(q, k, transpose_b=True)
        
        if mask is not None:
            nW = tf.shape(mask)[0]
            attn = tf.reshape(attn, (B_ // nW, nW, self.num_heads, N, N))
            attn += mask[:, None, :, :]
            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
        bias = tf.gather(self.relative_position_bias_table,
                         tf.reshape(self.relative_position_index, [-1]))
        bias = tf.reshape(bias, [N, N, self.num_heads])
        bias = tf.transpose(bias, [2, 0, 1])
        bias = tf.clip_by_value(bias, -5.0, 5.0)
        attn = attn + bias[None, ...]
        attn = tf.clip_by_value(attn, -10.0, 10.0)

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn, training=training)
        out  = tf.matmul(attn, v)
        out  = tf.transpose(out, (0, 2, 1, 3))
        out  = tf.reshape(out, (B_, N, C))
        out = self.proj(out)
        return self.proj_dropout(out, training=training)

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, input_resolution, num_heads=4,
                 window_size=4, shift_size=0,
                 attn_drop_rate=0., proj_drop_rate=0.,
                 mlp_drop_rate=0., drop_path_rate=0.1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads

        # Clamp window size to resolution
        H, W = input_resolution
        ws = min(window_size, H, W)
        self.window_size = ws
        self.shift_size = min(shift_size, ws // 2)

        # Layer norms and MLP
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate
        )
        self.attn_dropout = layers.Dropout(attn_drop_rate)
        self.proj_dropout = layers.Dropout(proj_drop_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = models.Sequential([
            layers.Dense(4 * dim),
            layers.Activation("gelu"),
            layers.Dropout(mlp_drop_rate),
            layers.Dense(dim),
            layers.Dropout(mlp_drop_rate),
        ])
        self.drop_path = DropPath(drop_prob=drop_path_rate)

        # Build attention mask if needed
        if self.shift_size > 0:
            img_mask = np.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(
                tf.convert_to_tensor(img_mask),
                (self.window_size, self.window_size)
            )
            mask_windows = tf.reshape(mask_windows, shape=(-1, self.window_size * self.window_size))
            attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
            attn_mask = tf.where(attn_mask != 0, -1e9, 0.0)
            self.attn_mask = tf.cast(attn_mask, tf.float32)
        else:
            self.attn_mask = None

    def call(self, x, training=None):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        shortcut = x
        x = self.norm1(x)
        # cyclic shift
        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        # partition windows
        x_windows = window_partition(x, (self.window_size, self.window_size))
        x_windows = tf.reshape(x_windows, shape=[-1, self.window_size * self.window_size, C])

        # attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = self.attn_dropout(attn_windows, training=training)
        attn_windows = tf.reshape(
            attn_windows,
            shape=[-1, self.window_size, self.window_size, C]
        )

        # merge windows
        x = window_reverse(attn_windows, (self.window_size, self.window_size), H, W)
        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])

        x = self.proj_dropout(x, training=training)
        x = shortcut + self.drop_path(x, training=training)
        x = self.norm2(x)
        x = self.mlp(x, training=training)
        x = shortcut + self.drop_path(x, training=training)
        return x

    def call(self, x, training=None):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        shortcut = x
        x = self.norm1(x)
        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1,2])
        x_windows = window_partition(x, (self.window_size, self.window_size))
        x_windows = tf.reshape(x_windows, shape=[-1, self.window_size * self.window_size, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = self.attn_dropout(attn_windows, training=training)
        attn_windows = tf.reshape(attn_windows, shape=[-1, self.window_size, self.window_size, C])
        x = window_reverse(attn_windows, (self.window_size, self.window_size), H, W)


        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1,2])
        x = self.proj_dropout(x, training=training)
        x = shortcut + self.drop_path(x, training=training)
        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x, training=training)
        x = shortcut2 + self.drop_path(x, training=training)
        return x

class PatchEmbedding(layers.Layer):
    def __init__(self, in_ch, embed_dim, patch_size):
        super().__init__()
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size,
                                  strides=patch_size, padding='valid')
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class PatchMerging(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.input_dim = dim
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.reduction = layers.Dense(2 * dim, use_bias=False)

    def call(self, x):
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, [B, H//2, 2, W//2, 2, C])
        x = tf.transpose(x, [0, 1, 3, 4, 2, 5])
        x = tf.reshape(x, [B, H//2, W//2, 4*C])
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpansion(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.input_dim = dim
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.expand = layers.Dense(2*dim, use_bias=False)

    def call(self, x):
        x = self.expand(x)
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, [B, H, W, 2, 2, C//4])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H*2, W*2, C//4])
        x = self.norm(x)
        return x

class FinalPatchExpansion(layers.Layer):
    def __init__(self, dim, up_factor):
        super().__init__()
        self.up_factor = up_factor
        self.norm      = layers.LayerNormalization(epsilon=1e-5)
        # up_factor**2 * dim, not hard‐coded 16*dim
        self.expand    = layers.Dense((up_factor**2) * dim, use_bias=False)

    def call(self, x):
        # x: [B, H, W, dim]
        x = self.expand(x)  # now has shape [B, H, W, up_factor**2 * dim]
        # get dynamic shape
        shape = tf.shape(x)
        B = shape[0]
        H = shape[1]
        W = shape[2]
        C = shape[3]  # this is up_factor**2 * original_dim

        # reshape into [B, H, W, up_factor, up_factor, C/(up_factor**2)]
        new_C = C // (self.up_factor**2)
        x = tf.reshape(
            x,
            [B,
             H,
             W,
             self.up_factor,
             self.up_factor,
             new_C]
        )
        # permute so we get [B, H, up_factor, W, up_factor, new_C]
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        # finally collapse to [B, H*up_factor, W*up_factor, new_C]
        x = tf.reshape(x, [B, H * self.up_factor, W * self.up_factor, new_C])

        return self.norm(x)


class SwinBlock(layers.Layer):
    def __init__(self, dims, ip_res, ss_size=3, num_heads=4):   
        super().__init__()
        self.swtb1 = SwinTransformerBlock(dim=dims, input_resolution=ip_res, num_heads=num_heads,)
        self.swtb2 = SwinTransformerBlock(dim=dims, input_resolution=ip_res,
                                         shift_size=ss_size, num_heads=num_heads)

    def call(self, x):
        x = self.swtb2(self.swtb1(x))
        return x

class Encoder(models.Model):
    def __init__(self, C, input_resolution, num_stages=3):
        super().__init__()
        H, W = input_resolution

        self.enc_swin_blocks = [
            keras.Sequential([
                SwinBlock(C, (H, W), num_heads=C//32),
                SwinBlock(C, (H, W), num_heads=C//32)
            ]),
            keras.Sequential([
                SwinBlock(2 * C, (H // 2, W // 2), num_heads=(2*C)//32),
                SwinBlock(2 * C, (H // 2, W // 2), num_heads=(2*C)//32)
            ]),
            keras.Sequential([
                SwinBlock(4 * C, (H // 4, W // 4), num_heads=(4*C)//32),
                SwinBlock(4 * C, (H // 4, W // 4), num_heads=(4*C)//32)
            ])
        ]

        self.enc_patch_merge_blocks = [
            PatchMerging(C),
            PatchMerging(2 * C),
            PatchMerging(4 * C)
        ]

    def call(self, x, training=None):
        skip_conn_ftrs = []
        for swin_blocks, patch_merger in zip(self.enc_swin_blocks, self.enc_patch_merge_blocks):
            x = swin_blocks(x, training=training)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs


class Decoder(models.Model):
    def __init__(self, C, input_resolution):
        super().__init__()
        H, W = input_resolution

        self.dec_swin_blocks = [
            keras.Sequential([
                SwinBlock(4 * C, (H // 4, W // 4), num_heads=(4*C)//32),
                SwinBlock(4 * C, (H // 4, W // 4), num_heads=(4*C)//32)
            ]),
            keras.Sequential([
                SwinBlock(2 * C, (H // 2, W // 2), num_heads=(2*C)//32),
                SwinBlock(2 * C, (H // 2, W // 2), num_heads=(2*C)//32)
            ]),
            keras.Sequential([
                SwinBlock(C, (H, W), num_heads=C//32),
                SwinBlock(C, (H, W), num_heads=C//32)
            ])
        ]

        self.dec_patch_expand_blocks = [
            PatchExpansion(8 * C),
            PatchExpansion(4 * C),
            PatchExpansion(2 * C)
        ]

        self.skip_conv_layers = [
            layers.Conv2D(4 * C, kernel_size=1, use_bias=False),
            layers.Conv2D(2 * C, kernel_size=1, use_bias=False),
            layers.Conv2D(C, kernel_size=1, use_bias=False)
        ]

    def call(self, x, encoder_features, training=None):
        for patch_expand, swin_blocks, enc_ftr, skip_conv in zip(
            self.dec_patch_expand_blocks,
            self.dec_swin_blocks,
            reversed(encoder_features),
            self.skip_conv_layers
        ):
            x = patch_expand(x)
            x = tf.concat([x, enc_ftr], axis=-1)
            x = skip_conv(x)
            x = swin_blocks(x, training=training)
        return x
    

class ConvStem(layers.Layer):
    def __init__(self, out_channels, kernel_size=7, stride=4, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=False
        )
        self.bn   = layers.BatchNormalization()
        self.act  = layers.Activation("gelu")

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x,    training=training)
        return self.act(x)


class SwinUNet(models.Model):
    def __init__(self, H, W, ch, C, num_class=1, num_blocks=3, patch_size=16):
        super().__init__()

        # conv stem: large kernel, stride to downsample once
        #self.stem = ConvStem(out_channels=C, kernel_size=7, stride=4, name="stem")

        self.patch_embed = PatchEmbedding(ch, C, patch_size)
        self.encoder = Encoder(C, (H//patch_size, W//patch_size))
        self.bottleneck = SwinBlock(C*(2**num_blocks),
                                   (H//(patch_size*(2**num_blocks)),
                                    W//(patch_size*(2**num_blocks))))
        self.decoder = Decoder(C, (H//patch_size, W//patch_size))
        # … in your SwinUNet __init__ …
        self.final_expansion = FinalPatchExpansion(C, up_factor=patch_size)

        self.head = layers.Conv2D(num_class, 1, activation='sigmoid')

    def call(self, x):
        x = self.patch_embed(x)
        x, skip_ftrs = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_ftrs)
        x = self.final_expansion(x)
        x = self.head(x)
        return x