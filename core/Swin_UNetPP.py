import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def window_partition(x, window_size):
    """Window partition function for Swin Transformer"""
    B, H, W, C = x.shape
    x = tf.reshape(x, [B, H // window_size[0], window_size[0],
                      W // window_size[1], window_size[1], C])
    windows = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, [-1, window_size[0], window_size[1], C])
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] // (H * W / window_size[0] / window_size[1]))
    x = tf.reshape(windows, [B, H // window_size[0], W // window_size[1],
                            window_size[0], window_size[1], -1])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, -1])
    return x

def get_relative_position_index(win_h, win_w):
    """Generate relative position index for windows"""
    coords = tf.meshgrid(tf.range(win_h), tf.range(win_w), indexing='ij')
    coords = tf.stack(coords)
    coords_flatten = tf.reshape(coords, [2, -1])
    
    # Reshape to ensure correct broadcasting dimensions
    coords_flatten = tf.reshape(coords_flatten, [2, 1, -1])
    
    # Calculate relative coordinates with correct broadcasting
    relative_coords = coords_flatten[:, tf.newaxis, :] - coords_flatten[tf.newaxis, :, :]
    
    # Reshape to final form
    relative_coords = tf.reshape(relative_coords, [-1])
    
    return tf.cast(relative_coords, tf.int32)

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size):
        super().__init__()
        # Convert scalar to tuple if necessary
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = 4
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        
        # Now window_size[0] and window_size[1] will work correctly
        self.relative_position_bias_table = self.add_weight(
            shape=(2 * window_size[0] - 1, 2 * window_size[1] - 1, self.num_heads),
            initializer=tf.random_normal_initializer(stddev=0.02),
            trainable=True,
            name="relative_position_bias"
        )
        indices = get_relative_position_index(window_size[0], window_size[1])
        self.relative_position_bias_table = tf.Variable(indices, trainable=False)
        self.qkv = layers.Dense(dim * 3)
        self.proj = layers.Dense(dim)
    
    def _get_rel_pos_bias(self):
        relative_position_bias = tf.gather_nd(
            self.relative_position_bias_table,
            self.relative_position_index
        )
        return tf.expand_dims(tf.transpose(relative_position_bias, [2, 0, 1]), 0)
    
    def call(self, x, mask=None):
        B_, N, C = tf.shape(x)
        
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, -1])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv
        
        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        attn = attn + self._get_rel_pos_bias()
        
        if mask is not None:
            num_win = tf.shape(mask)[0]
            attn = tf.reshape(attn, [-1, num_win, self.num_heads, N, N]) + \
                   tf.expand_dims(mask, axis=1)
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
        
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B_, N, -1])
        x = self.proj(x)
        return x

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, input_resolution, window_size=7, shift_size=0):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.window_area = window_size * window_size
        
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim=dim, window_size=window_size)
        
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            layers.Dense(4 * dim),
            layers.GELU(),
            layers.Dense(dim)
        ])
        
        if shift_size:
            H, W = input_resolution
            img_mask = np.zeros((1, H, W, 1))
            cnt = 0
            for h in (slice(0, -window_size),
                     slice(-window_size, -shift_size),
                     slice(-shift_size, None)):
                for w in (slice(0, -window_size),
                         slice(-window_size, -shift_size),
                         slice(-shift_size, None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = tf.reshape(mask_windows, [-1, window_area])
            attn_mask = mask_windows[:, tf.newaxis] - mask_windows[:, tf.newaxis, :]
            attn_mask = tf.where(attn_mask != 0, float('-inf'), float(0))
            self.attn_mask = tf.Variable(attn_mask, trainable=False)
        else:
            self.attn_mask = None
    
    def _attn(self, x):
        B, H, W, C = tf.shape(x)
        
        if self.shift_size:
            shifted_x = tf.roll(x, shifts=(-self.shift_size, -self.shift_size), 
                              axis=(1, 2))
        else:
            shifted_x = x
            
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(x_windows, [-1, self.window_area, C])
        
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, 
                                               self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        if self.shift_size:
            x = tf.roll(shifted_x, shifts=(self.shift_size, self.shift_size), 
                       axis=(1, 2))
        else:
            x = shifted_x
            
        return x
    
    def call(self, x):
        B, H, W, C = tf.shape(x)
        
        x = x + self._attn(self.norm1(x))
        x = tf.reshape(x, [B, -1, C])
        x = x + self.mlp(self.norm2(x))
        x = tf.reshape(x, [B, H, W, C])
        
        return x

class PatchEmbedding(layers.Layer):
    def __init__(self, in_ch, num_feat, patch_size):
        super().__init__()
        self.conv = layers.Conv2D(
            filters=num_feat,
            kernel_size=patch_size,
            strides=patch_size,
            padding='valid'
        )
    
    def call(self, x):
        return tf.transpose(self.conv(x), [0, 2, 3, 1])

class PatchMerging(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.reduction = layers.Dense(2 * dim, use_bias=False)
    
    def call(self, x):
        B, H, W, C = tf.shape(x)
        x = tf.reshape(x, [B, H // 2, 2, W // 2, 2, C])
        x = tf.transpose(x, [0, 1, 3, 4, 2, 5])
        x = tf.reshape(x, [B, H // 2, W // 2, 4 * C])
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpansion(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.expand = layers.Dense(2 * dim, use_bias=False)
    
    def call(self, x):
        x = self.expand(x)
        B, H, W, C = tf.shape(x)
        x = tf.reshape(x, [B, H, W, 2, 2, C//4])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H*2, W*2, C//4])
        x = self.norm(x)
        return x

class FinalPatchExpansion(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = layers.LayerNormalization(epsilon=1e-5)
        self.expand = layers.Dense(16 * dim, use_bias=False)
    
    def call(self, x):
        x = self.expand(x)
        B, H, W, C = tf.shape(x)
        x = tf.reshape(x, [B, H, W, 4, 4, C//16])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H*4, W*4, C//16])
        x = self.norm(x)
        return x

class SwinBlock(layers.Layer):
    def __init__(self, dims, ip_res, ss_size=3):
        super().__init__()
        self.swtb1 = SwinTransformerBlock(dim=dims, input_resolution=ip_res)
        self.swtb2 = SwinTransformerBlock(
            dim=dims,
            input_resolution=ip_res,
            shift_size=ss_size
        )
    
    def call(self, x):
        return self.swtb2(self.swtb1(x))

class Encoder(models.Model):
    def __init__(self, C, partitioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partitioned_ip_res
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
        for swin_block, patch_merger in zip(
                self.enc_swin_blocks, 
                self.enc_patch_merge_blocks):
            x = swin_block(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs

class Decoder(models.Model):
    def __init__(self, C, partitioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partitioned_ip_res
        self.dec_swin_blocks = [
            SwinBlock(4*C, (H//4, W//4)),
            SwinBlock(2*C, (H//2, W//2)),
            SwinBlock(C, (H, W))
        ]
        
        self.dec_patch_expand_blocks = [
            PatchExpansion(8*C),
            PatchExpansion(4*C),
            PatchExpansion(2*C)
        ]
        
        self.skip_conn_concat = [
            layers.Dense(4*C, use_bias=False),
            layers.Dense(2*C, use_bias=False),
            layers.Dense(C, use_bias=False)
        ]
    
    def call(self, x, encoder_features):
        for patch_expand, swin_block, enc_ftr, linear_concatter in zip(
                self.dec_patch_expand_blocks,
                self.dec_swin_blocks,
                encoder_features[::-1],
                self.skip_conn_concat):
            x = patch_expand(x)
            x = tf.concat([x, enc_ftr], axis=-1)
            x = linear_concatter(x)
            x = swin_block(x)
        return x

class SwinUNetPP(models.Model):
    def __init__(self, input_shape, num_classes, window_size=(7, 7), embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.patch_size = 4
        self.window_size = window_size  # Now accepts a tuple
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.patch_embed = PatchEmbedding(input_shape[-1], embed_dim, self.patch_size)
        self.encoder = Encoder(embed_dim, (input_shape[0]//self.patch_size, input_shape[1]//self.patch_size), num_blocks=len(depths))
        self.bottleneck = SwinBlock(embed_dim*(2**len(depths)),
                                   (input_shape[0]//(self.patch_size*(2**len(depths))),
                                    input_shape[1]//(self.patch_size*(2**len(depths)))),
                                   ss_size=window_size[0])  # Use window_size[0] here
        self.decoder = Decoder(embed_dim, (input_shape[0]//self.patch_size, input_shape[1]//self.patch_size), num_blocks=len(depths))
        self.final_expansion = FinalPatchExpansion(embed_dim)
        self.head = layers.Conv2D(num_classes, 1, padding='same')

    def call(self, x):
        x = self.patch_embed(x)
        x, skip_ftrs = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_ftrs[::-1])
        x = self.final_expansion(x)
        x = tf.transpose(x, [0, 3, 1, 2])
        x = self.head(x)
        return x