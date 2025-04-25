import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def window_partition(x, window_size):
    """
    Partition batch of images into windows.
    Args:
        x: tensor of shape (B, H, W, C)
        window_size: tuple of integers (window_height, window_width)
    Returns:
        windows: tensor of shape (num_windows*B, window_height, window_width, C)
    """
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    x = tf.reshape(x, [B,
        H // window_size[0], window_size[0],
        W // window_size[1], window_size[1],
        C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])  # [B, num_win_h, num_win_w, wh, ww, C]
    windows = tf.reshape(x, [-1, window_size[0], window_size[1], C])  # [num_windows*B, wh, ww, C]
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse windows to original batch of images.
    Args:
        windows: tensor of shape (num_windows*B, window_height, window_width, C)
        window_size: tuple (window_height, window_width)
        H: original image height (int or tf.Tensor)
        W: original image width (int or tf.Tensor)
    Returns:
        x: tensor of shape (B, H, W, C)
    """
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

def get_relative_position_index(win_h, win_w):
    """Get pair-wise relative position index for each token inside the window."""
    coords = np.meshgrid(np.arange(win_h), np.arange(win_w), indexing='ij')
    coords_flatten = np.stack([coord.flatten() for coord in coords])
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = np.transpose(relative_coords, [1, 2, 0])
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    relative_coords = relative_coords.astype(np.float32)
    tf.debugging.assert_all_finite(tf.constant(relative_coords),
                                 "NaN/Inf detected in relative position indices")
    return tf.constant(relative_coords.sum(-1))

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = 4
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = tf.Variable(
            initial_value=tf.zeros(
                shape=(2 * window_size[0] - 1) ** 2,
                dtype=tf.float32
            )
        )
        self.relative_position_index = get_relative_position_index(
            window_size[0],
            window_size[1]
        )
        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)
        tf.random.normal(
            self.relative_position_bias_table.shape,
            stddev=0.02,
            dtype=tf.float32
        )

    def _get_rel_pos_bias(self):
        """Get relative position bias."""
        relative_position_indices = tf.cast(
            self.relative_position_index,
            tf.int64
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(relative_position_indices, [-1])
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            [self.window_area, self.window_area, -1]
        )
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1])
        return relative_position_bias[tf.newaxis, ...]

    def call(self, x, mask=None):
        """Forward pass for window attention with optional mask."""
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)  # (B_, N, 3 * C)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B_, num_heads, N, head_dim)
        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)  # (B_, num_heads, N, N)
        rel_pos_bias = self._get_rel_pos_bias()  # (1, num_heads, N, N)
        attn = attn + rel_pos_bias
        if mask is not None:
            num_windows = tf.shape(mask)[0]
            attn = tf.reshape(attn, [-1, num_windows, self.num_heads, N, N])
            attn = attn + mask[tf.newaxis, :, tf.newaxis, :, :]
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
        attn = tf.nn.softmax(attn, axis=-1)
        attn = tf.matmul(attn, v)  # (B_, num_heads, N, head_dim)
        x = tf.transpose(attn, [0, 2, 1, 3])  # (B_, N, num_heads, head_dim)
        x = tf.reshape(x, [B_, N, C])         # (B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, input_resolution, window_size=4, shift_size=0):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.window_area = window_size * window_size
        self.dim = dim
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim=dim, window_size=(window_size, window_size))
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            layers.Dense(4 * dim),
            layers.Activation('gelu'),
            layers.Dense(dim)
        ])
        if shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            cnt = 0
            for h in (slice(0, -window_size),
                     slice(-window_size, -shift_size),
                     slice(-shift_size, None)):
                for w in (slice(0, -window_size),
                         slice(-window_size, -shift_size),
                         slice(-shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, (window_size, window_size))
            mask_windows = tf.reshape(mask_windows, [-1, self.window_area])
            attn_mask = mask_windows[:, tf.newaxis] - mask_windows[:, tf.newaxis, :]
            attn_mask = tf.where(attn_mask != 0, tf.constant(float('-inf'), dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
            attn_mask = tf.cast(attn_mask, dtype=tf.float32)
            tf.debugging.assert_all_finite(attn_mask, "NaN/Inf in attention mask")
            self.attn_mask = attn_mask
        else:
            self.attn_mask = None

    def _attn(self, x):
        """Attention mechanism."""
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x
        x_windows = window_partition(shifted_x, (self.window_size, self.window_size))
        x_windows = tf.reshape(x_windows, [-1, self.window_area, C])
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        tf.debugging.assert_all_finite(attn_windows, "NaN/Inf in attention output")
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, (self.window_size, self.window_size), H, W)
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, [B, H, W, C])
        return x

    def call(self, x):
        """Forward pass."""
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        x = x + self._attn(self.norm1(x))
        x = tf.reshape(x, [B, H * W, C])
        x = x + self.mlp(self.norm2(x))
        x = tf.reshape(x, [B, H, W, C])
        return x

class PatchEmbedding(layers.Layer):
    def __init__(self, in_ch, num_feat, patch_size):
        super().__init__()
        self.conv = layers.Conv2D(num_feat, patch_size, strides=patch_size)

    def call(self, x):
        x = self.conv(x)
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
    def __init__(self, dim):
        super().__init__()
        self.input_dim = dim
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.expand = layers.Dense(16*dim, use_bias=False)

    def call(self, x):
        x = self.expand(x)
        shape = tf.shape(x)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, [B, H, W, 4, 4, C//16])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H*4, W*4, C//16])
        x = self.norm(x)
        return x

class SwinBlock(layers.Layer):
    def __init__(self, dims, ip_res, ss_size=3):
        super().__init__()
        self.swtb1 = SwinTransformerBlock(dim=dims, input_resolution=ip_res)
        self.swtb2 = SwinTransformerBlock(dim=dims, input_resolution=ip_res,
                                         shift_size=ss_size)

    def call(self, x):
        x = self.swtb2(self.swtb1(x))
        return x

class Encoder(models.Model):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partioned_ip_res
        self.enc_swin_blocks = [
            SwinBlock(C, (H, W)),
            SwinBlock(2*C, (H//2, W//2)),
            SwinBlock(4*C, (H//4, W//4))
        ]
        self.enc_patch_merge_blocks = [
            PatchMerging(C),
            PatchMerging(2*C),
            PatchMerging(4*C)
        ]

    def call(self, x):
        skip_conn_ftrs = []
        for swin_block, patch_merger in zip(self.enc_swin_blocks,
                                           self.enc_patch_merge_blocks):
            x = swin_block(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs

class Decoder(models.Model):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partioned_ip_res
        self.dec_swin_blocks = [
            SwinBlock(4*C, (H//4, W//4)),      # 384 channels
            SwinBlock(2*C, (H//2, W//2)),      # 192 channels
            SwinBlock(C, (H, W))               # 96 channels
        ]
        self.dec_patch_expand_blocks = [
            PatchExpansion(8*C),    # 768 -> 384
            PatchExpansion(4*C),    # 384 -> 192
            PatchExpansion(2*C)     # 192 -> 96
        ]
        self.skip_conn_concat = [
            layers.Dense(4*C, use_bias=False),    # 768 -> 384
            layers.Dense(2*C, use_bias=False),    # 384 -> 192
            layers.Dense(C, use_bias=False)       # 192 -> 96
        ]

    def call(self, x, encoder_features):
        for patch_expand, swin_block, enc_ftr, linear_concatter in zip(
            self.dec_patch_expand_blocks,
            self.dec_swin_blocks,
            reversed(encoder_features),
            self.skip_conn_concat):
            x = patch_expand(x)
            x = tf.concat([x, enc_ftr], axis=-1)
            x = linear_concatter(x)
            x = swin_block(x)
        return x

class SwinUNet(models.Model):
    def __init__(self, H, W, ch, C, num_class=1, num_blocks=3, patch_size=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(ch, C, patch_size)
        self.encoder = Encoder(C, (H//patch_size, W//patch_size), num_blocks)
        self.bottleneck = SwinBlock(C*(2**num_blocks),
                                   (H//(patch_size*(2**num_blocks)),
                                    W//(patch_size*(2**num_blocks))))
        self.decoder = Decoder(C, (H//patch_size, W//patch_size), num_blocks)
        self.final_expansion = FinalPatchExpansion(C)
        self.head = layers.Conv2D(num_class, 1, activation='sigmoid')

    def call(self, x):
        x = self.patch_embed(x)
        x, skip_ftrs = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_ftrs)
        x = self.final_expansion(x)
        x = self.head(x)
        return x