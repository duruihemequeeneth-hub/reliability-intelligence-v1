import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.cmapss_loader import load_cmapss, add_rul_labels, drop_constant_sensors

def prepare_rul_data(df, max_cycles=None):
    """
    Prepare data for RUL prediction.
    Features: all sensor readings + operating settings
    Target: RUL (capped at 125 cycles as per standard practice)
    """
    # Drop non-feature columns
    exclude_cols = ['unit_number', 'rul']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['rul'].copy()
    
    # Cap RUL at 125 cycles (standard for C-MAPSS)
    y = y.clip(upper=125)
    
    return X, y, feature_cols

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model for RUL prediction with hyperparameter tuning.
    Updated for XGBoost 3.3.0+ - early_stopping_rounds moved to constructor.
    """
    # XGBoost parameters (optimized for C-MAPSS)
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20  # Now in constructor for v3.3+
    )
    
    # Train model
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return xgb_model

def train_random_forest_model(X_train, y_train):
    """
    Train Random Forest model as baseline comparison.
    """
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate model performance and return metrics.
    """
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate prediction accuracy within ±10 cycles
    within_10 = np.mean(np.abs(y_test - y_pred) <= 10) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy_within_10': within_10,
        'predictions': y_pred
    }

def plot_rul_predictions(y_test, y_pred, model_name, save_path='assets/rul_predictions.png'):
    """
    Visualize RUL predictions vs actual values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Scatter plot: Actual vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual RUL (cycles)')
    ax1.set_ylabel('Predicted RUL (cycles)')
    ax1.set_title(f'{model_name}: Actual vs Predicted RUL')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual distribution
    ax2 = axes[1]
    residuals = y_test - y_pred
    ax2.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residual (Actual - Predicted)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{model_name}: Residual Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    return fig

def plot_feature_importance(model, feature_names, save_path='assets/feature_importance.png'):
    """
    Plot feature importance from XGBoost model.
    """
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot top 15 features
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = importance.head(15)
    ax.barh(top_features['feature'], top_features['importance'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 15 Most Important Features for RUL Prediction')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    return fig, importance

def predict_rul_with_confidence(model, X, n_iterations=100):
    """
    Predict RUL with confidence intervals using bootstrapping.
    """
    # Bootstrap predictions
    predictions = []
    for i in range(n_iterations):
        # Sample with replacement
        idx = np.random.choice(len(X), len(X), replace=True)
        X_sample = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx]
        pred = model.predict(X_sample)
        predictions.append(pred)
    
    # Convert to array
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean_pred = np.mean(predictions, axis=0)
    lower_ci = np.percentile(predictions, 10, axis=0)  # 80% CI
    upper_ci = np.percentile(predictions, 90, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    return {
        'mean': mean_pred,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'std': std_pred,
        'all_predictions': predictions
    }

def plot_rul_with_confidence(y_test, rul_ci, engine_ids=None, save_path='assets/rul_confidence.png'):
    """
    Plot RUL predictions with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by actual RUL for cleaner visualization
    n_samples = len(y_test)
    x = np.arange(n_samples)
    
    # Plot confidence bands
    ax.fill_between(x, rul_ci['lower_ci'], rul_ci['upper_ci'], 
                    alpha=0.3, color='blue', label='80% Confidence Interval')
    ax.plot(x, rul_ci['mean'], 'b-', label='Predicted RUL', linewidth=2)
    ax.scatter(x, y_test, color='red', s=20, alpha=0.5, label='Actual RUL')
    
    ax.set_xlabel('Test Sample Index')
    ax.set_ylabel('RUL (cycles)')
    ax.set_title('RUL Predictions with 80% Confidence Intervals')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    return fig

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("REMAINING USEFUL LIFE (RUL) PREDICTION")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading C-MAPSS FD001 data...")
    df = load_cmapss('FD001', 'train')
    df = add_rul_labels(df)
    df = drop_constant_sensors(df)
    
    # Prepare data
    print("\n2. Preparing data for RUL prediction...")
    X, y, feature_cols = prepare_rul_data(df)
    print(f"   Features: {len(feature_cols)}")
    print(f"   Total samples: {len(X)}")
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Split train into train/val for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost model
    print("\n4. Training XGBoost model...")
    try:
        xgb_model = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
        use_xgboost = True
        print("   XGBoost training successful!")
    except Exception as e:
        print(f"   XGBoost failed ({e}), falling back to Random Forest...")
        xgb_model = train_random_forest_model(X_train_scaled, y_train)
        use_xgboost = False
        print("   Random Forest training successful!")
    
    # Evaluate XGBoost
    print("\n5. Evaluating model...")
    xgb_results = evaluate_model(xgb_model, X_test_scaled, y_test, 'XGBoost' if use_xgboost else 'Random Forest')
    print(f"   MAE: {xgb_results['MAE']:.2f} cycles")
    print(f"   RMSE: {xgb_results['RMSE']:.2f} cycles")
    print(f"   R² Score: {xgb_results['R2']:.3f}")
    print(f"   Accuracy within ±10 cycles: {xgb_results['Accuracy_within_10']:.1f}%")
    
    # Plot predictions
    print("\n6. Generating visualization plots...")
    plot_rul_predictions(y_test, xgb_results['predictions'], 'XGBoost' if use_xgboost else 'Random Forest', 'assets/rul_predictions.png')
    
    # Feature importance (only for XGBoost)
    if use_xgboost:
        plot_feature_importance(xgb_model, feature_cols, 'assets/feature_importance.png')
    else:
        print("   Skipping feature importance (Random Forest doesn't provide as interpretable importance)")
    
    # Predict with confidence intervals
    print("\n7. Calculating confidence intervals for predictions...")
    # Use a subset of test data for demonstration
    X_test_subset = X_test_scaled[:50]
    y_test_subset = y_test.iloc[:50] if isinstance(y_test, pd.Series) else y_test[:50]
    
    rul_ci = predict_rul_with_confidence(xgb_model, X_test_subset, n_iterations=100)
    plot_rul_with_confidence(y_test_subset, rul_ci, save_path='assets/rul_confidence.png')
    
    # Save model and scaler
    print("\n8. Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgboost_rul_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("   Model saved to models/xgboost_rul_model.pkl")
    print("   Scaler saved to models/scaler.pkl")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual_rul': y_test,
        'predicted_rul': xgb_results['predictions'],
        'residual': y_test - xgb_results['predictions']
    })
    predictions_df.to_csv('data/processed/rul_predictions.csv', index=False)
    print("   Predictions saved to data/processed/rul_predictions.csv")
    
    print("\n" + "=" * 60)
    print("RUL PREDICTION COMPLETE")
    print("=" * 60)
