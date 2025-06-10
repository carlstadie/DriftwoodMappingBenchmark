import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

unet_loss_dir = "/isipd/projects/p_planetdw/data/methods_test/logs/unet_samples"

metrics = ['loss', 'specificity', 'sensitivity', 'IoU', 'f1_score', 'Hausdorff_distance']
output_dir = "/isipd/projects/p_planetdw/data/methods_test/logs/unet_samples"

def read_metrics_as_array(directory, metrics):
    files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])
    data_list = []
    
    metric_names = []
    for metric in metrics:
        metric_names.append(metric)
        metric_names.append('val_' + metric)

    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        file_data = []

        for name in metric_names:
            if name in df.columns:
                file_data.append(df[name].values)
            else:
                # Fill with NaNs if column is missing
                file_data.append(np.full(len(df), np.nan))
        
        # Transpose so shape is (epochs, metrics)
        file_data = np.stack(file_data, axis=1)  # shape: (epochs, num_metrics)
        data_list.append(file_data)

    # Convert to a 3D array: (files, epochs, metrics)
    data_array = np.stack(data_list, axis=0)

    # Build lookup dict
    lookup = {name: idx for idx, name in enumerate(metric_names)}

    return data_array, lookup, files

data_array, metric_lookup, file_names = read_metrics_as_array(unet_loss_dir, metrics)

print(f"Data shape: {data_array.shape}")



def plot_losses(loss_array, metrics, metric_lookup, output_dir):
    epochs = loss_array.shape[1]
    num_metrics = len(metrics)

    plt.figure(figsize=(10, 10))  # Square layout
    
    for i, metric in enumerate(metrics):
        plt.subplot(num_metrics, 1, i + 1)

        # Training metric
        for j in range(loss_array.shape[0]):
            plt.plot(range(epochs), loss_array[j, :, metric_lookup[metric]], color='lightblue', linewidth=1)

        train_mean = np.nanmean(loss_array[:, :, metric_lookup[metric]], axis=0)
        
        # Validation metric
        val_metric = 'val_' + metric
        if val_metric in metric_lookup:
            for j in range(loss_array.shape[0]):
                plt.plot(range(epochs), loss_array[j, :, metric_lookup[val_metric]], color='peachpuff', linewidth=1)

            val_mean = np.nanmean(loss_array[:, :, metric_lookup[val_metric]], axis=0)
            plt.plot(range(epochs), train_mean, color='tab:blue', label=f'train', linewidth=2)
            plt.plot(range(epochs), val_mean, color='tab:orange', label=f'val', linewidth=2)

        plt.title(metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.gca().set_aspect('auto')  # Square plot per metric (approx)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'losses_plot.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


plot_losses(data_array, metrics, metric_lookup, output_dir)
    


