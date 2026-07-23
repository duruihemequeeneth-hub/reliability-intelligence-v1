import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def compare_models(X_train, y_train, X_test, y_test, models_dict=None):
    """
    Compare multiple models for RUL prediction.
    
    Parameters:
    - X_train, y_train, X_test, y_test: Training and test data
    - models_dict: Dictionary of models to compare (default: common models)
    
    Returns:
    - DataFrame with comparison results
    """
    if models_dict is None:
        models_dict = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
            'SVM': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
    
    results = []
    predictions = {}
    
    for name, model in models_dict.items():
        try:
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation (5-fold)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results.append({
                'model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2,
                'CV_R²_Mean': np.mean(cv_scores),
                'CV_R²_Std': np.std(cv_scores)
            })
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            results.append({
                'model': name,
                'MAE': np.nan,
                'RMSE': np.nan,
                'R²': np.nan,
                'CV_R²_Mean': np.nan,
                'CV_R²_Std': np.nan
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('R²', ascending=False)
    
    return results_df, predictions

def plot_model_comparison(results_df, predictions_dict, y_test, save_path='assets/model_comparison.png'):
    """
    Visualize model comparison results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Performance comparison (bar chart)
    ax1 = axes[0, 0]
    models = results_df['model'].values
    r2_scores = results_df['R²'].values
    colors = ['green' if r2 >= 0.8 else 'orange' if r2 >= 0.7 else 'red' for r2 in r2_scores]
    ax1.bar(models, r2_scores, color=colors)
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good Performance')
    ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Acceptable')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance Comparison')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE comparison
    ax2 = axes[0, 1]
    mae_values = results_df['MAE'].values
    ax2.bar(models, mae_values, color='red')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('MAE (cycles)')
    ax2.set_title('Mean Absolute Error Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cross-validation scores
    ax3 = axes[1, 0]
    cv_mean = results_df['CV_R²_Mean'].values
    cv_std = results_df['CV_R²_Std'].values
    ax3.bar(models, cv_mean, yerr=cv_std, capsize=5, color='blue')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Cross-Validation R² Score')
    ax3.set_title('Cross-Validation Performance (5-fold)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Predictions comparison (scatter plot)
    ax4 = axes[1, 1]
    best_model = results_df.iloc[0]['model']
    y_pred_best = predictions_dict[best_model]
    ax4.scatter(y_test, y_pred_best, alpha=0.5, s=20, label=f'{best_model} (R²={results_df.iloc[0]["R²"]:.3f})')
    
    # Also plot another model for comparison
    if len(results_df) > 1:
        second_model = results_df.iloc[1]['model']
        y_pred_second = predictions_dict[second_model]
        ax4.scatter(y_test, y_pred_second, alpha=0.3, s=20, label=f'{second_model} (R²={results_df.iloc[1]["R²"]:.3f})')
    
    ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect Prediction')
    ax4.set_xlabel('Actual RUL (cycles)')
    ax4.set_ylabel('Predicted RUL (cycles)')
    ax4.set_title('Predictions: Best vs. Second Best')
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
    print("MODEL COMPARISON DEMONSTRATION")
    print("=" * 60)
    print("\nThis module compares multiple models for RUL prediction.")
    print("Integrate this with your RUL predictor to validate model selection.")
