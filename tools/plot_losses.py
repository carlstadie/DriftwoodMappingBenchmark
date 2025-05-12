import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

def extract_metrics_from_events(log_dir):
    """
    Extract training metrics from TensorFlow event logs into a DataFrame.
    Handles profile-empty files and searches subdirectories.
    
    Args:
        log_dir (str): Path to the directory containing events.out.tfevents.* files
        
    Returns:
        tuple: Training DataFrame, Validation DataFrame
    """
    train_data = []
    val_data = []
    
    # Verify log directory exists
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # Get all event files, including in subdirectories
    event_files = tf.io.gfile.glob(f"{log_dir}/events.out.tfevents.*")
    if not event_files:
        # Try subdirectories
        event_files = tf.io.gfile.glob(f"{log_dir}/**/events.out.tfevents.*")
    
    print(f"Found {len(event_files)} event files:")
    for file_path in event_files:
        print(f"  - {file_path}")
    
    # Process each event file
    for file_path in event_files:
        try:
            print(f"\nProcessing file: {file_path}")
            # Use tf.compat.v1.summary.Summary() for more robust parsing
            for e in tf.compat.v1.train.summary_iterator(file_path):
                if e.summary:
                    for v in e.summary.value:
                        if v.tensor:
                            metric_name = v.tag
                            value = float(tf.make_ndarray(v.tensor))
                            step = int(e.step)
                            
                            print(f"  Step {step}: {metric_name} = {value}")
                            
                            # Determine if this is training or validation
                            if 'val_' in metric_name:
                                val_data.append({
                                    'metric': metric_name.replace('val_', ''),
                                    'value': value,
                                    'step': step
                                })
                            else:
                                train_data.append({
                                    'metric': metric_name,
                                    'value': value,
                                    'step': step
                                })
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Create DataFrames
    df_train = pd.DataFrame(train_data)
    df_val = pd.DataFrame(val_data)
    
    return df_train, df_val

# Replace with your actual log directory path
log_dir = "/isipd/projects/p_planetdw/data/methods_test/logs/UNet/20250509-1600_U"

# Extract metrics
try:
    df_train, df_val = extract_metrics_from_events(log_dir)
    
    print("\nFirst few rows of training metrics:")
    print(df_train.head())
    print("\nFirst few rows of validation metrics:")
    print(df_val.head())
    
    if df_train.empty:
        print("\nWarning: Training DataFrame is empty")
    if df_val.empty:
        print("\nWarning: Validation DataFrame is empty")
except Exception as e:
    print(f"Error: {str(e)}")