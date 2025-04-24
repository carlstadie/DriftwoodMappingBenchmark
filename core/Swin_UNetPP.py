import tensorflow as tf
import numpy as np
import math

def window_partition(x, window_size):
    B, H, W, C = tf.shape(x)
    
    # Reshape to [B, H//window_size[0], window_size[0], W//window_size[1], window_size[1], C]
    x = tf.reshape(x, [
        B,
        H // window_size[0],
        window_size[0],
        W // window_size[1],
        window_size[1],
        C
    ])
    
    # Transpose dimensions and flatten
    windows = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, [
        -1,
        window_size[0],
        window_size[1],
        C
    ])
    
    return windows

def create_attention_mask(H, W, window_size, shift_size):
    img_mask = tf.zeros((1, H, W, 1), dtype=tf.int32)

    h_slices = [
        (0, H - window_size[0]),
        (H - window_size[0], H - shift_size[0]),
        (H - shift_size[0], H)
    ]
    w_slices = [
        (0, W - window_size[1]),
        (W - window_size[1], W - shift_size[1]),
        (W - shift_size[1], W)
    ]

    cnt = 0
    for h_start, h_end in h_slices:
        for w_start, w_end in w_slices:
            mask_patch = tf.ones((1, h_end - h_start, w_end - w_start, 1), dtype=tf.int32) * cnt
            paddings = [[0, 0],
                        [h_start, H - h_end],
                        [w_start, W - w_end],
                        [0, 0]]
            padded_patch = tf.pad(mask_patch, paddings, constant_values=0)
            img_mask += padded_patch
            cnt += 1

    return img_mask


def window_reverse(windows, window_size, H, W):
    # Get number of channels
    C = tf.shape(windows)[-1]
    
    # Reshape windows
    x = tf.reshape(windows, [
        -1,
        H // window_size[0],
        W // window_size[1],
        window_size[0],
        window_size[1],
        C
    ])
    
    # Transpose dimensions
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    
    # Reshape back to original spatial dimensions
    x = tf.reshape(x, [-1, H, W, C])
    
    return x

def get_relative_position_index(win_h: int, win_w: int):
    coords = tf.stack(tf.meshgrid(
        tf.range(win_h),
        tf.range(win_w),
        indexing='ij'
    ))  # (2, win_h, win_w)
    
    coords_flatten = tf.reshape(coords, [2, -1])  # (2, win_h * win_w)
    
    relative_coords = coords_flatten[:, :, tf.newaxis] - coords_flatten[:, tf.newaxis, :]  # (2, N, N)
    relative_coords = tf.transpose(relative_coords, [1, 2, 0])  # (N, N, 2)

    # Shift to all positive indices
    relative_coords += [win_h - 1, win_w - 1]  # ensures indices are >= 0

    # Map 2D relative coords to 1D index
    relative_position_index = relative_coords[:, :, 0] * (2 * win_w - 1) + relative_coords[:, :, 1]

    return tf.cast(relative_position_index, tf.int32)


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.num_heads = 4
        head_dim = dim // self.num_heads

        self.relative_position_bias_table = self.add_weight(
            shape=(2 * window_size[0] - 1, 2 * window_size[1] - 1, self.num_heads),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name='relative_position_bias'
        )

        # ✅ Store as plain tensor, not a weight
        self.relative_position_index = get_relative_position_index(window_size[0], window_size[1])

        self.scale = head_dim ** -0.5
        self.qkv = tf.keras.layers.Dense(dim * 3)
        self.proj = tf.keras.layers.Dense(dim)

        
    def _get_rel_pos_bias(self):
        # Flatten bias table to shape [num_entries, num_heads]
        bias_flat = tf.reshape(self.relative_position_bias_table, [-1, self.num_heads])  # [num_positions, heads]

        # Use relative_position_index (shape [window_area, window_area]) to gather from flat table
        bias = tf.gather(bias_flat, self.relative_position_index)  # shape: [window_area, window_area, num_heads]

        # Rearrange to [1, num_heads, N, N]
        bias = tf.transpose(bias, [2, 0, 1])  # [heads, N, N]
        bias = tf.expand_dims(bias, axis=0)  # [1, heads, N, N]
        return bias

    
    def call(self, x, mask=None):
        B_, N, C = tf.shape(x)
        
        # Compute QKV
        qkv = self.qkv(x)  # [B_, N, 3 * dim]
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, -1])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3, B_, heads, N, head_dim]
        q, k, v = tf.unstack(qkv, axis=0)  # now each is [B_, heads, N, head_dim]
        
        # Scale queries
        q = q * self.scale
        
        # Compute attention
        attn = tf.matmul(q, k, transpose_b=True)
        attn = attn + self._get_rel_pos_bias()
        
        # Apply mask if provided
        if mask is not None:
            num_win = tf.shape(mask)[0]
            attn = tf.reshape(attn, [-1, num_win, self.num_heads, N, N])
            attn = attn + mask[:, tf.newaxis, tf.newaxis, :, :]
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
        
        # Apply softmax
        attn = tf.nn.softmax(attn, axis=-1)
        
        # Compute weighted sum
        x = tf.matmul(attn, v)  # (B_, heads, N, head_dim)

        # Transpose and merge heads
        x = tf.transpose(x, [0, 2, 1, 3])  # (B_, N, heads, head_dim)
        x = tf.reshape(x, [B_, N, -1])     # (B_, N, C)

        # Final projection
        x = self.proj(x)

        
        return x
    
class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, window_size=8, shift_size=0):
        super().__init__()
        self.input_resolution = input_resolution
        window_size = (window_size, window_size)
        shift_size = (shift_size, shift_size)
        self.window_size = window_size
        self.shift_size = shift_size
        self.window_area = window_size[0] * window_size[1]
        
        # Layer normalization
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1.001e-5)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1.001e-5)
        
        # Attention
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size
        )
        
        # MLP
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * dim, activation='gelu'),
            tf.keras.layers.Dense(dim)
        ])
        
        # Calculate attention mask for shifted window attention

        H, W = input_resolution

        if shift_size[0] > 0 or shift_size[1] > 0:
            H_pad = math.ceil(H / window_size[0]) * window_size[0]
            W_pad = math.ceil(W / window_size[1]) * window_size[1]
            img_mask = create_attention_mask(H_pad, W_pad, window_size, shift_size)

            mask_windows = window_partition(img_mask, window_size)
            mask_windows = tf.reshape(mask_windows, [-1, self.window_area])
            attn_mask = mask_windows[:, tf.newaxis, :] - mask_windows[:, :, tf.newaxis]
            attn_mask = tf.where(attn_mask != 0, -100.0, 0.0)

            self.attn_mask = self.add_weight(
                name='attn_mask',
                shape=attn_mask.shape,
                initializer=tf.keras.initializers.Constant(attn_mask.numpy()),  # Use .numpy() here
                trainable=False
            )
        else:
            self.attn_mask = None

    
    def _attn(self, x):
        # Get dynamic input shape
        input_shape = tf.shape(x)
        B, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]


        # Save original H, W for unpadding later
        pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]

        Hp, Wp = H + pad_h, W + pad_w

        # Pad if needed
        x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

        # Cyclic shift
        if self.shift_size != (0, 0):
            shifted_x = tf.roll(x, shift=[-self.shift_size[0], -self.shift_size[1]], axis=[1, 2])
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)  # [num_windows*B, win_h, win_w, C]
        x_windows = tf.reshape(x_windows, [-1, self.window_area, C])  # [num_windows*B, win_area, C]

        # Apply attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = tf.reshape(attn_windows, [-1, self.window_size[0], self.window_size[1], C])
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size != (0, 0):
            x = tf.roll(shifted_x, shift=[self.shift_size[0], self.shift_size[1]], axis=[1, 2])
        else:
            x = shifted_x

        # Remove padding
        x = x[:, :H, :W, :]

        return x

    def call(self, x):
        B, H, W, C = tf.shape(x)

        attn_out = self._attn(self.norm1(x))  # already returns (B, H, W, C)
        x = x + attn_out

        x = tf.reshape(x, [B, -1, C])
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        x = tf.reshape(x, [B, H, W, C])
        return x


    
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_feat=128, patch_size=4):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_feat,
            kernel_size=patch_size,
            strides=patch_size
        )
        
    def call(self, X):
        # Input shape: (batch_size, 256, 256, in_channels)
        # Output shape: (batch_size, 64, 64, num_feat)
        return tf.transpose(self.conv(X), [0, 2, 3, 1])

class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1.001e-5)
        self.reduction = tf.keras.layers.Dense(2 * dim, use_bias=False)
    
    def call(self, x):
        B, H, W, C = tf.shape(x)
        x = tf.reshape(x, [B, H // 2, 2, W // 2, 2, C])
        x = tf.transpose(x, [0, 1, 3, 4, 2, 5])
        x = tf.reshape(x, [B, H // 2, W // 2, 4 * C])
        x = self.norm(x)
        x = self.reduction(x)
        return x

class PatchExpansion(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1.001e-5)
        self.expand = tf.keras.layers.Dense(2 * dim, use_bias=False)
    
    def call(self, x):
        x = self.expand(x)
        B, H, W, C = tf.shape(x)
        x = tf.reshape(x, [B, H, W, 2, 2, C // 4])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H * 2, W * 2, C // 4])
        x = self.norm(x)
        return x

class FinalPatchExpansion(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1.001e-5)
        self.expand = tf.keras.layers.Dense(16 * dim, use_bias=False)
    
    def call(self, x):
        x = self.expand(x)
        B, H, W, C = tf.shape(x)
        x = tf.reshape(x, [B, H, W, 4, 4, C // 16])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [B, H * 4, W * 4, C // 16])
        x = self.norm(x)
        return x
    
class SwinBlock(tf.keras.layers.Layer):
    def __init__(self, dims, ip_res, ss_size=3):
        super().__init__()
        self.swtb1 = SwinTransformerBlock(
            dim=dims,
            input_resolution=ip_res
        )
        self.swtb2 = SwinTransformerBlock(
            dim=dims,
            input_resolution=ip_res,
            shift_size=ss_size
        )
    
    def call(self, x):
        return self.swtb2(self.swtb1(x))
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partioned_ip_res[0], partioned_ip_res[1]
        
        # Create Swin blocks with increasing channel dimensions
        self.enc_swin_blocks = [
            SwinBlock(C, (H, W)),
            SwinBlock(2*C, (H//2, W//2)),
            SwinBlock(4*C, (H//4, W//4))
        ]
        
        # Create patch merging blocks
        self.enc_patch_merge_blocks = [
            PatchMerging(C),
            PatchMerging(2*C),
            PatchMerging(4*C)
        ]
    
    def call(self, x):
        skip_conn_ftrs = []
        for swin_block, patch_merger in zip(self.enc_swin_blocks, self.enc_patch_merge_blocks):
            x = swin_block(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs
    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partioned_ip_res[0], partioned_ip_res[1]
        
        # Create Swin blocks with decreasing channel dimensions
        self.dec_swin_blocks = [
            SwinBlock(4*C, (H//4, W//4)),
            SwinBlock(2*C, (H//2, W//2)),
            SwinBlock(C, (H, W))
        ]
        
        # Create patch expansion blocks
        self.dec_patch_expand_blocks = [
            PatchExpansion(8*C),
            PatchExpansion(4*C),
            PatchExpansion(2*C)
        ]
        
        # Create skip connection concatenation layers
        self.skip_conn_concat = [
            tf.keras.layers.Dense(4*C, use_bias=False),
            tf.keras.layers.Dense(2*C, use_bias=False),
            tf.keras.layers.Dense(C, use_bias=False)
        ]
    
    def call(self, x, encoder_features):
        for patch_expand, swin_block, enc_ftr, linear_concatter in zip(
            self.dec_patch_expand_blocks, 
            self.dec_swin_blocks, 
            encoder_features,
            self.skip_conn_concat
        ):
            x = patch_expand(x)
            x = tf.concat([x, enc_ftr], axis=-1)
            x = linear_concatter(x)
            x = swin_block(x)
        return x
    
class SwinUNet(tf.keras.Model):
    def __init__(self, H=256, W=256, in_channels=4, C=128, num_class=1, num_blocks=3, patch_size=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=in_channels, num_feat=C, patch_size=patch_size)
        self.encoder = Encoder(C, (H//patch_size, W//patch_size), num_blocks)
        self.bottleneck = SwinBlock(C*(2**num_blocks),
                                (H//(patch_size*(2**num_blocks)),
                                    W//(patch_size*(2**num_blocks))))
        self.decoder = Decoder(C, (H//patch_size, W//patch_size), num_blocks)
        self.final_expansion = FinalPatchExpansion(C)
        self.head = tf.keras.layers.Conv2D(
            filters=num_class,
            kernel_size=1,
            padding='same'
        )
    
    def call(self, x):
        x = self.patch_embed(x)
        x, skip_ftrs = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_ftrs[::-1])
        x = self.final_expansion(x)
        x = self.head(tf.transpose(x, [0, 3, 1, 2]))
        return x