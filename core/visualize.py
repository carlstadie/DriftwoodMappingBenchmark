# core/visualize.py

"""
Small utilities for quick image display using Matplotlib.

The functions here are intentionally minimal; richer visualization during
training and validation is handled by the TensorBoard-centric utilities.

Actually, I dont think I use this anymore. 
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon  # imported for API parity


def display_images(img, titles=None, cmap=None, norm=None, interpolation=None):
    """
    Display a batch of single-channel images.

    Args:
        img: Array-like in shape (B, H, W, C).
        titles: Optional list of titles (length B*C).
        cmap: Optional Matplotlib colormap name or instance.
        norm: Optional Normalize instance to map values to colors.
        interpolation: Optional interpolation name passed to imshow.
    """
    cols = img.shape[-1]
    rows = img.shape[0]
    titles = titles if titles is not None else [""] * (rows * cols)

    plt.figure(figsize=(14, 14 * rows // cols))
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows, cols, (i * cols) + j + 1)
            plt.axis("off")
            plt.imshow(img[i, ..., j], cmap=cmap, norm=norm, interpolation=interpolation)
    # plt.show()
