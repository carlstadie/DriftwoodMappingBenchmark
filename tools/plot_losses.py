import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- load your data ---
df = pd.read_csv(
    "/isipd/projects/p_planetdw/data/methods_test/logs/20250513-1740_MACS_test_metrics.csv",
    sep=","
)
val_loss   = df["val_loss"].values
train_loss = df["loss"].values

# --- generate synthetic losses ---
def genetrate_synthetic_losses(val_loss, train_loss, sample_size=100):
    syn_val_loss, syn_train_loss = [], []
    for _ in range(sample_size):
        shift = np.random.normal(-0.01, 0.01, size=val_loss.shape)
        syn_val_loss.append(val_loss   + shift)
        syn_train_loss.append(train_loss + shift)
    return np.array(syn_val_loss), np.array(syn_train_loss)




def plot_losses(
    syn_train_loss: np.ndarray,
    syn_val_loss:   np.ndarray,
    save_path:      str = None,
    show:           bool = True,
    figsize:        tuple = (12, 5)
):
    """
    Plots synthetic training & validation losses with their means in a 1×2 subplot.

    Parameters
    ----------
    syn_train_loss : np.ndarray
        Array of shape (n_runs, n_epochs) for training losses.
    syn_val_loss : np.ndarray
        Array of shape (n_runs, n_epochs) for validation losses.
    save_path : str, optional
        If given, the figure is saved to this path.
    show : bool
        If True, calls plt.show() at the end.
    figsize : tuple
        Size of the overall figure (width, height).
    """
    # compute mean and std
    mean_train = syn_train_loss #np.mean(syn_train_loss, axis=0)
    std_train  = np.std(syn_train_loss,  axis=0)
    mean_val   = syn_val_loss #np.mean(syn_val_loss,   axis=0)
    std_val    = np.std(syn_val_loss,    axis=0)

    # setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Training Loss ±σ
    epochs = np.arange(mean_train.size)
    ax1.plot(epochs, mean_train, color='C0', lw=2, label='Mean Train Loss')
    """ax1.fill_between(
        epochs,
        mean_train - std_train,
        mean_train + std_train,
        color='C0', alpha=0.3,
        label='±1σ'
    )"""
    ax1.set_title("Training Loss")
    ax1.set_ylim(0, 1.)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Validation Loss ±σ
    ax2.plot(epochs, mean_val, color='C1', lw=2, label='Mean Val Loss')
    """ax2.fill_between(
        epochs,
        mean_val - std_val,
        mean_val + std_val,
        color='C1', alpha=0.3,
        label='±1σ'
    )"""
    ax2.set_title("Validation Loss")
    ax2.set_ylim(0, 1.)
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

syn_val_loss, syn_train_loss = genetrate_synthetic_losses(val_loss, train_loss, sample_size=100)

plot_losses(
    train_loss,
    val_loss,
    save_path="/isipd/projects/p_planetdw/data/methods_test/logs/combined_losses_swin.png"
)

