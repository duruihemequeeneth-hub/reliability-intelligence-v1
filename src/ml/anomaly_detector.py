import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.cmapss_loader import load_cmapss, add_rul_labels, drop_constant_sensors

def prepare_sensor_data(df, engine_id=None):
    """
    Prepare sensor data for anomaly detection.
    If engine_id is provided, filter for that specific engine.
    """
    if engine_id is not None:
        df_engine = df[df['unit_number'] == engine_id].copy()
    else:
        df_engine = df.copy()
    
    # Identify sensor columns (exclude non-sensor columns)
    exclude_cols = ['unit_number', 'time_in_cycles', 'op_setting_1', 
                   'op_setting_2', 'op_setting_3', 'rul']
    sensor_cols = [col for col in df_engine.columns if col not in exclude_cols]
    
    # Prepare feature matrix
    X = df_engine[sensor_cols].values
    
    return X, sensor_cols, df_engine

def detect_anomalies_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Detect anomalies using Isolation Forest.
    contamination: expected proportion of outliers (0.1 = 10%)
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100
    )
    predictions = iso_forest.fit_predict(X_scaled)
    
    # Convert to binary: 1 = normal, -1 = anomaly
    anomaly_labels = (predictions == -1).astype(int)
    anomaly_scores = iso_forest.score_samples(X_scaled)
    
    return {
        'anomaly_labels': anomaly_labels,
        'anomaly_scores': anomaly_scores,
        'model': iso_forest,
        'scaler': scaler
    }

def calculate_degradation_trend(df, engine_id, window_size=10):
    """
    Calculate degradation trend using moving average slope.
    """
    df_engine = df[df['unit_number'] == engine_id].copy()
    
    # Use sensor_2 (total temperature at fan inlet) as proxy for degradation
    # In practice, you'd use the most sensitive sensor
    sensor_col = 'sensor_2'
    
    # Calculate rolling average
    df_engine['rolling_mean'] = df_engine[sensor_col].rolling(window=window_size).mean()
    
    # Calculate slope (rate of change) over window
    df_engine['slope'] = df_engine['rolling_mean'].diff(window_size) / window_size
    
    return df_engine

def plot_anomaly_detection(df, engine_id, results, save_path='assets/anomaly_detection.png'):
    """
    Visualize anomaly detection results for a specific engine.
    """
    # Get engine data
    df_engine = df[df['unit_number'] == engine_id].copy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Sensor trend with anomalies highlighted
    ax1 = axes[0]
    sensor_col = 'sensor_2'  # Most sensitive sensor
    ax1.plot(df_engine['time_in_cycles'], df_engine[sensor_col], 'b-', alpha=0.7, label='Sensor 2')
    
    # Highlight anomalies
    anomaly_indices = np.where(results['anomaly_labels'] == 1)[0]
    if len(anomaly_indices) > 0:
        anomaly_times = df_engine.iloc[anomaly_indices]['time_in_cycles'].values
        anomaly_values = df_engine.iloc[anomaly_indices][sensor_col].values
        ax1.scatter(anomaly_times, anomaly_values, color='red', s=50, zorder=5, label='Anomalies')
    
    ax1.set_title(f'Engine {engine_id}: Sensor Trend with Anomalies', fontsize=14)
    ax1.set_xlabel('Time (cycles)')
    ax1.set_ylabel('Sensor Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Anomaly scores
    ax2 = axes[1]
    ax2.plot(df_engine['time_in_cycles'], results['anomaly_scores'], 'g-', alpha=0.7)
    ax2.axhline(y=np.percentile(results['anomaly_scores'], 10), color='red', linestyle='--', alpha=0.5, label='Anomaly threshold')
    ax2.fill_between(df_engine['time_in_cycles'], 
                     results['anomaly_scores'], 
                     np.percentile(results['anomaly_scores'], 10),
                     where=(results['anomaly_scores'] < np.percentile(results['anomaly_scores'], 10)),
                     color='red', alpha=0.3)
    ax2.set_title(f'Engine {engine_id}: Anomaly Scores', fontsize=14)
    ax2.set_xlabel('Time (cycles)')
    ax2.set_ylabel('Anomaly Score (lower = more anomalous)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Degradation trend (FIXED)
    ax3 = axes[2]
    df_trend = calculate_degradation_trend(df, engine_id)
    ax3.plot(df_trend['time_in_cycles'], df_trend['rolling_mean'], 'b-', linewidth=2, label='Rolling Mean')
    
    # Add trend line - fixed alignment
    valid_mask = df_trend['rolling_mean'].notna()
    x_clean = df_trend.loc[valid_mask, 'time_in_cycles'].values
    y_clean = df_trend.loc[valid_mask, 'rolling_mean'].values
    
    if len(x_clean) > 1 and len(y_clean) > 1:
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        ax3.plot(df_trend['time_in_cycles'], p(df_trend['time_in_cycles']), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.4f})')
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for trend line', transform=ax3.transAxes, ha='center')
    
    ax3.set_title(f'Engine {engine_id}: Degradation Trend', fontsize=14)
    ax3.set_xlabel('Time (cycles)')
    ax3.set_ylabel('Sensor Value (rolling mean)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure assets directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    return fig

def analyze_anomalies_across_fleet(df, contamination=0.1):
    """
    Analyze anomalies across all engines and return summary statistics.
    """
    # Prepare data
    X, sensor_cols, _ = prepare_sensor_data(df)
    
    # Detect anomalies
    results = detect_anomalies_isolation_forest(X, contamination)
    
    # Add anomaly labels back to dataframe
    df['anomaly'] = results['anomaly_labels']
    
    # Summary statistics by engine
    anomaly_summary = df.groupby('unit_number').agg(
        total_rows=('time_in_cycles', 'count'),
        anomalies=('anomaly', 'sum'),
        anomaly_pct=('anomaly', lambda x: (x.sum() / len(x)) * 100)
    ).reset_index()
    
    return anomaly_summary, results

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("HEALTH MONITORING & ANOMALY DETECTION")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading C-MAPSS FD001 data...")
    df = load_cmapss('FD001', 'train')
    df = add_rul_labels(df)
    df = drop_constant_sensors(df)
    
    # Sample a few engines for demonstration
    sample_engines = [1, 50, 100]
    
    print("\n2. Detecting anomalies for sample engines...")
    
    for engine_id in sample_engines:
        print(f"\n   Analyzing Engine {engine_id}:")
        
        # Prepare data for this engine
        X, sensor_cols, df_engine = prepare_sensor_data(df, engine_id)
        
        # Detect anomalies
        results = detect_anomalies_isolation_forest(X, contamination=0.15)
        
        # Count anomalies
        n_anomalies = results['anomaly_labels'].sum()
        total_points = len(results['anomaly_labels'])
        anomaly_pct = (n_anomalies / total_points) * 100
        
        print(f"   Total data points: {total_points}")
        print(f"   Anomalies detected: {n_anomalies}")
        print(f"   Anomaly percentage: {anomaly_pct:.1f}%")
        
        # Plot results
        plot_anomaly_detection(df, engine_id, results, f'assets/anomaly_engine_{engine_id}.png')
    
    # Fleet-wide analysis
    print("\n3. Fleet-wide anomaly analysis...")
    anomaly_summary, fleet_results = analyze_anomalies_across_fleet(df, contamination=0.1)
    
    print(f"   Fleet average anomaly rate: {anomaly_summary['anomaly_pct'].mean():.1f}%")
    print(f"   Engines with >20% anomalies: {len(anomaly_summary[anomaly_summary['anomaly_pct'] > 20])}")
    print(f"   Most anomalous engine: {anomaly_summary.loc[anomaly_summary['anomaly_pct'].idxmax(), 'unit_number']} "
          f"({anomaly_summary['anomaly_pct'].max():.1f}% anomalies)")
    
    # Save summary
    anomaly_summary.to_csv('data/processed/anomaly_summary.csv', index=False)
    print("\n   Anomaly summary saved to data/processed/anomaly_summary.csv")
    
    print("\n" + "=" * 60)
    print("HEALTH MONITORING COMPLETE")
    print("=" * 60)
