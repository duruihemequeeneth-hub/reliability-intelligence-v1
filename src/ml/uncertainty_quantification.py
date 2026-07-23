import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def monte_carlo_rul_prediction(model, X, n_simulations=1000, confidence_level=0.95):
    """
    Perform Monte Carlo simulation for RUL prediction uncertainty.
    
    Parameters:
    - model: Trained XGBoost model
    - X: Feature matrix (scaled)
    - n_simulations: Number of Monte Carlo runs
    - confidence_level: Confidence interval level (e.g., 0.95 for 95% CI)
    
    Returns:
    - Dict with mean, percentiles, and full distribution
    """
    # Add Gaussian noise to features (simulate sensor uncertainty)
    noise_scale = 0.05  # 5% sensor noise
    
    predictions = []
    for _ in range(n_simulations):
        # Add random noise to X
        X_noisy = X + np.random.normal(0, noise_scale, X.shape)
        pred = model.predict(X_noisy)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    # Percentiles for confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_ci = np.percentile(predictions, lower_percentile, axis=0)
    upper_ci = np.percentile(predictions, upper_percentile, axis=0)
    
    # Calculate probability of failure before a given time
    failure_probabilities = []
    horizons = [10, 20, 30, 50, 75, 100]
    for horizon in horizons:
        prob = np.mean(predictions < horizon, axis=0)
        failure_probabilities.append({
            'horizon': horizon,
            'probability': np.mean(prob)
        })
    
    return {
        'mean': mean_pred,
        'std': std_pred,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'confidence_level': confidence_level,
        'all_predictions': predictions,
        'failure_probabilities': failure_probabilities
    }

def plot_uncertainty_analysis(y_test, uncertainty_results, save_path='assets/uncertainty_analysis.png'):
    """
    Visualize uncertainty quantification results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Prediction distribution for a single sample
    ax1 = axes[0, 0]
    sample_predictions = uncertainty_results['all_predictions'][:, 0]
    ax1.hist(sample_predictions, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(uncertainty_results['mean'][0], color='red', linestyle='--', label=f"Mean: {uncertainty_results['mean'][0]:.1f}")
    ax1.axvline(uncertainty_results['lower_ci'][0], color='orange', linestyle=':', label=f"Lower CI ({uncertainty_results['confidence_level']*100:.0f}%)")
    ax1.axvline(uncertainty_results['upper_ci'][0], color='orange', linestyle=':', label=f"Upper CI ({uncertainty_results['confidence_level']*100:.0f}%)")
    ax1.set_xlabel('Predicted RUL (cycles)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Monte Carlo Prediction Distribution (Sample)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence bands over time
    ax2 = axes[0, 1]
    n_samples = len(y_test)
    x = np.arange(n_samples)
    
    # Sort by actual RUL for cleaner visualization
    sorted_idx = np.argsort(y_test)
    y_test_sorted = y_test[sorted_idx]
    mean_sorted = uncertainty_results['mean'][sorted_idx]
    lower_sorted = uncertainty_results['lower_ci'][sorted_idx]
    upper_sorted = uncertainty_results['upper_ci'][sorted_idx]
    
    ax2.fill_between(x, lower_sorted, upper_sorted, alpha=0.3, color='blue', label=f'{uncertainty_results["confidence_level"]*100:.0f}% CI')
    ax2.plot(x, mean_sorted, 'b-', label='Predicted Mean', linewidth=2)
    ax2.scatter(x, y_test_sorted, color='red', s=10, alpha=0.5, label='Actual RUL')
    ax2.set_xlabel('Test Sample Index (sorted by RUL)')
    ax2.set_ylabel('RUL (cycles)')
    ax2.set_title('Confidence Bands for Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Failure probability over time
    ax3 = axes[1, 0]
    horizons = [10, 20, 30, 50, 75, 100]
    probs = [uncertainty_results['failure_probabilities'][i]['probability'] for i in range(len(horizons))]
    ax3.plot(horizons, probs, 'bo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Time Horizon (cycles)')
    ax3.set_ylabel('Failure Probability')
    ax3.set_title('Cumulative Failure Probability')
    ax3.grid(True, alpha=0.3)
    
    # 4. Prediction standard deviation distribution
    ax4 = axes[1, 1]
    ax4.hist(uncertainty_results['std'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.axvline(np.mean(uncertainty_results['std']), color='red', linestyle='--', label=f"Mean: {np.mean(uncertainty_results['std']):.2f}")
    ax4.set_xlabel('Prediction Standard Deviation')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Uncertainty Distribution Across Samples')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    return fig

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("UNCERTAINTY QUANTIFICATION DEMONSTRATION")
    print("=" * 60)
    print("\nThis module provides Monte Carlo simulation for RUL predictions.")
    print("It estimates prediction uncertainty and failure probabilities.")
    print("Integrate this with your RUL predictor for enhanced decision support.")
