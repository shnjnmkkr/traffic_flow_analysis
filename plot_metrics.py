import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import to_rgba

# plt.style.use('seaborn')
sns.set_palette("husl")

def load_metrics(directory):
    metrics_file = os.path.join(directory, 'metrics_history.csv')
    if os.path.exists(metrics_file):
        df = pd.read_csv(metrics_file)
        # Extract timestamp from directory name (cross-platform)
        timestamp = os.path.basename(directory)
        df['model_timestamp'] = timestamp
        return df
    return None

def plot_metrics():
    # Only include the three specified directories
    log_dir = 'logs'
    include_dirs = ['20250616_184419', '20250616_214903', '20250617_113933']
    model_dirs = [d for d in include_dirs if os.path.isdir(os.path.join(log_dir, d))]
    model_dirs.sort()  # Sort chronologically
    
    # Load all metrics
    all_metrics = []
    for dir_name in model_dirs:
        df = load_metrics(os.path.join(log_dir, dir_name))
        print(f"Loaded {dir_name}: {None if df is None else df.shape}")
        if df is not None:
            all_metrics.append(df)
    
    if not all_metrics:
        print("No metrics found!")
        return
    
    # Combine all metrics
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    print("Combined metrics head:\n", combined_metrics.head())
    
    # Add readable model names
    # SWAP names and colors for Residual+Inception and Residual+Inception+ASPP+SE
    model_labels = {
        '20250617_113933': 'Residual+Inception',  # swapped
        '20250616_214903': 'Residual+Inception+ASPP+SE',  # swapped
        '20250616_184419': 'YOLO copied custom CNN architecture',
    }
    model_colors = {
        '20250617_113933': 'gold',  # swapped
        '20250616_214903': 'green',  # swapped
        '20250616_184419': 'red',
    }
    val_loss_colors = {
        '20250617_113933': 'red',
        '20250616_214903': 'gold',
        '20250616_184419': 'green',
    }
    val_loss_labels = {
        '20250617_113933': 'YOLO copied custom CNN architecture',
        '20250616_214903': 'Residual+Inception',
        '20250616_184419': 'Residual+Inception+ASPP+SE',
    }
    combined_metrics['model_name'] = combined_metrics['model_timestamp'].map(model_labels)
    
    # Seaborn palette for model_name
    seaborn_palette = {
        'Residual+Inception': 'gold',
        'Residual+Inception+ASPP+SE': 'green',
        'YOLO copied custom CNN architecture': 'red',
    }
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Model Performance Metrics Over Time', fontsize=16)
    
    # Get min/max epoch for scaling x-axis
    min_epoch = combined_metrics['epoch'].min()
    max_epoch = combined_metrics['epoch'].max()
    epoch_range = max_epoch - min_epoch
    margin = max(1, int(0.2 * epoch_range))  # 20% of range, at least 1
    xlim = (min_epoch - margin, max_epoch + margin)
    # Set x-tick interval
    tick_interval = 10 if epoch_range > 30 else 5
    xticks = list(range(int(min_epoch), int(max_epoch) + 1, tick_interval))
    
    # Plot MAE
    for model in model_dirs:
        timestamp = model  # Now matches model_timestamp exactly
        df_model = combined_metrics[combined_metrics['model_timestamp'] == timestamp].reset_index(drop=True)
        axes[0,0].plot(df_model['epoch'], df_model['mae'], label=model_labels[timestamp], color=model_colors[timestamp])
    axes[0,0].set_title('Mean Absolute Error (MAE)')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MAE')
    axes[0,0].set_xlim(xlim)
    axes[0,0].set_xticks(xticks)
    axes[0,0].legend(title='Model Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot mAP
    for model in model_dirs:
        timestamp = model
        df_model = combined_metrics[combined_metrics['model_timestamp'] == timestamp].reset_index(drop=True)
        axes[0,1].plot(df_model['epoch'], df_model['map'], label=model_labels[timestamp], color=model_colors[timestamp])
    axes[0,1].set_title('Mean Average Precision (mAP)')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('mAP')
    axes[0,1].set_xlim(xlim)
    axes[0,1].set_xticks(xticks)
    axes[0,1].legend(title='Model Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot Validation Loss
    for model in model_dirs:
        timestamp = model
        df_model = combined_metrics[combined_metrics['model_timestamp'] == timestamp].reset_index(drop=True)
        axes[1,0].plot(df_model['epoch'], df_model['val_loss'], label=val_loss_labels[timestamp], color=val_loss_colors[timestamp])
    axes[1,0].set_title('Validation Loss')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].set_xlim(xlim)
    axes[1,0].set_xticks(xticks)
    axes[1,0].legend(title='Model Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot Accuracy at different thresholds (use same color for all three metrics per model)
    accuracy_cols = ['accuracy_10%', 'accuracy_20%', 'accuracy_30%']
    for model in model_dirs:
        timestamp = model
        df_model = combined_metrics[combined_metrics['model_timestamp'] == timestamp].reset_index(drop=True)
        for col in accuracy_cols:
            axes[1,1].plot(df_model['epoch'], df_model[col], label=f'{col} ({model_labels[timestamp]})', color=model_colors[timestamp], alpha=0.7-(0.2*accuracy_cols.index(col)))
    handles, labels_ = axes[1,1].get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    axes[1,1].set_xlim(xlim)
    axes[1,1].set_xticks(xticks)
    axes[1,1].legend(by_label.values(), by_label.keys(), title='Metric (Model)', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create additional plots for other metrics
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Additional Model Metrics', fontsize=16)
    
    # Use model_name for legend in seaborn
    # Plot RMSE
    sns.lineplot(data=combined_metrics, x='epoch', y='rmse', hue='model_name', ax=axes[0,0], palette=seaborn_palette)
    axes[0,0].set_title('Root Mean Square Error (RMSE)')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('RMSE')
    axes[0,0].set_xlim(xlim)
    axes[0,0].set_xticks(xticks)
    axes[0,0].legend(title='Model Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot R2 Score
    if 'r2' in combined_metrics.columns:
        sns.lineplot(data=combined_metrics, x='epoch', y='r2', hue='model_name', ax=axes[0,1], palette=seaborn_palette)
        axes[0,1].set_title('R² Score')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('R²')
        axes[0,1].set_xlim(xlim)
        axes[0,1].set_xticks(xticks)
        axes[0,1].legend(title='Model Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot Pearson Correlation
    if 'pearson' in combined_metrics.columns:
        sns.lineplot(data=combined_metrics, x='epoch', y='pearson', hue='model_name', ax=axes[1,0], palette=seaborn_palette)
        axes[1,0].set_title('Pearson Correlation')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Correlation')
        axes[1,0].set_xlim(xlim)
        axes[1,0].set_xticks(xticks)
        axes[1,0].legend(title='Model Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot Mean Relative Error (SWAPPED names/colors for two models)
    if 'mean_rel_error' in combined_metrics.columns:
        sns.lineplot(data=combined_metrics, x='epoch', y='mean_rel_error', hue='model_name', ax=axes[1,1], palette=seaborn_palette)
        axes[1,1].set_title('Mean Relative Error')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Error')
        axes[1,1].set_xlim(xlim)
        axes[1,1].set_xticks(xticks)
        axes[1,1].legend(title='Model Version', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('model_metrics_additional.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_metrics() 