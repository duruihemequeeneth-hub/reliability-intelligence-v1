import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")

def analyze_feature_importance_shap(model, X, feature_names, n_samples=100):
    """
    Perform SHAP analysis for model interpretability.
    
    Parameters:
    - model: Trained XGBoost model
    - X: Feature matrix
    - feature_names: List of feature names
    - n_samples: Number of samples for SHAP explanation (use subset for speed)
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Install with: pip install shap")
        return None
    
    # Sample data for SHAP (too many samples is slow)
    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Feature importance (mean absolute SHAP values)
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    return {
        'shap_values': shap_values,
        'shap_importance': shap_importance,
        'explainer': explainer,
        'X_sample': X_sample
    }

def plot_shap_summary(shap_results, feature_names, save_path='assets/shap_summary.png'):
    """
    Visualize SHAP summary plot.
    """
    if not SHAP_AVAILABLE or shap_results is None:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by importance
    shap_importance = shap_results['shap_importance']
    
    # Get top 15 features
    top_features = shap_importance.head(15)['feature'].values
    
    # Create summary plot data
    shap_values = shap_results['shap_values']
    X_sample = shap_results['X_sample']
    
    # Manually create a summary-like plot
    # Calculate mean SHAP value per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]
    
    # Plot top features
    top_idx = sorted_idx[:15]
    y_pos = np.arange(len(top_idx))
    
    # Use the SHAP library's summary plot if available, otherwise manual
    try:
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
        plt.show()
        return fig
    except Exception as e:
        print(f"Error generating SHAP summary plot: {e}")
        return None

def plot_feature_importance_comparison(shap_importance, xgb_importance, feature_names, save_path='assets/feature_importance_comparison.png'):
    """
    Compare SHAP importance with XGBoost feature importance.
    """
    if shap_importance is None:
        return None
    
    # Create combined dataframe
    xgb_importance_df = pd.DataFrame({
        'feature': feature_names,
        'xgb_importance': xgb_importance
    })
    
    combined = shap_importance.merge(xgb_importance_df, on='feature')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # SHAP Importance
    ax1 = axes[0]
    top_shap = shap_importance.head(10)
    ax1.barh(top_shap['feature'], top_shap['shap_importance'], color='blue')
    ax1.set_xlabel('Mean |SHAP Value|')
    ax1.set_title('SHAP Feature Importance')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # XGBoost Importance
    ax2 = axes[1]
    top_xgb = xgb_importance_df.sort_values('xgb_importance', ascending=False).head(10)
    ax2.barh(top_xgb['feature'], top_xgb['xgb_importance'], color='green')
    ax2.set_xlabel('XGBoost Feature Importance')
    ax2.set_title('XGBoost Feature Importance')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
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
    print("EXPLAINABLE AI (SHAP) DEMONSTRATION")
    print("=" * 60)
    print("\nThis module provides SHAP analysis for model interpretability.")
    print("It helps explain which features drive RUL predictions.")
    print("Install SHAP with: pip install shap")
