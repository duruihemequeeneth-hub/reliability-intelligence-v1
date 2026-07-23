import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def simulate_vacuum_pump(
    n_assets=20,
    n_days=365,
    failure_threshold=0.3,
    decay_rate=0.005,
    sampling_freq='H'  # Hourly sampling
):
    """
    Simulate vacuum pump degradation data for semiconductor manufacturing.
    
    Parameters:
    - n_assets: Number of pumps to simulate
    - n_days: Simulation duration in days
    - failure_threshold: Health index below which pump fails
    - decay_rate: Rate of degradation per day
    - sampling_freq: Sampling frequency ('H'=hourly, 'D'=daily)
    
    Returns:
    - DataFrame with simulated sensor data
    """
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=n_days),
        end=datetime.now(),
        freq=sampling_freq
    )
    
    all_data = []
    
    for asset_id in range(1, n_assets + 1):
        # Initial health varies per asset (manufacturing variation)
        initial_health = np.random.uniform(0.95, 1.0)
        
        # Each asset has slightly different degradation rate
        asset_decay_rate = decay_rate * np.random.uniform(0.8, 1.2)
        
        for i, timestamp in enumerate(timestamps):
            # Base health degradation (exponential)
            health = initial_health * np.exp(-asset_decay_rate * i)
            
            # Add noise (sensor noise + process variation)
            health += np.random.normal(0, 0.01)
            
            # Add random shocks (occasional spikes)
            if np.random.random() < 0.01:  # 1% chance of shock
                health -= np.random.uniform(0.05, 0.15)
            
            # Clamp to realistic range
            health = np.clip(health, 0, 1)
            
            # Determine if asset has failed
            failed = health < failure_threshold
            
            # Generate sensor readings (correlated with health)
            pressure = 100 - (1 - health) * 50 + np.random.normal(0, 5)
            temperature = 25 + (1 - health) * 30 + np.random.normal(0, 2)
            motor_current = 10 + (1 - health) * 5 + np.random.normal(0, 0.5)
            vibration = (1 - health) * 3 + np.random.normal(0, 0.1)
            
            # RUL (in hours)
            if health > failure_threshold:
                # Estimate time to failure based on current degradation rate
                hours_to_failure = (health - failure_threshold) / (asset_decay_rate * health)
                rul_hours = min(hours_to_failure, 24 * 30)  # Cap at 30 days
            else:
                rul_hours = 0
            
            all_data.append({
                'asset_id': asset_id,
                'timestamp': timestamp,
                'time_hours': i,
                'health_index': health,
                'failure': failed,
                'pressure_pa': pressure,
                'temperature_c': temperature,
                'motor_current_a': motor_current,
                'vibration_mm_s': vibration,
                'rul_hours': rul_hours
            })
    
    df = pd.DataFrame(all_data)
    return df

def simulate_cooling_system(
    n_assets=10,
    n_days=180,
    sampling_freq='D'  # Daily sampling
):
    """
    Simulate cooling system degradation data.
    """
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=n_days),
        end=datetime.now(),
        freq=sampling_freq
    )
    
    all_data = []
    
    for asset_id in range(1, n_assets + 1):
        # Initial efficiency
        initial_efficiency = np.random.uniform(0.9, 1.0)
        
        # Linear degradation (cooling systems often degrade linearly)
        degradation_rate = np.random.uniform(0.001, 0.003)
        failure_threshold = 0.6
        
        for i, timestamp in enumerate(timestamps):
            # Efficiency decreases linearly
            efficiency = initial_efficiency - degradation_rate * i
            
            # Add noise
            efficiency += np.random.normal(0, 0.02)
            efficiency = np.clip(efficiency, 0, 1)
            
            # Sensor readings
            flow_rate = 100 * efficiency + np.random.normal(0, 2)
            coolant_temp = 20 + (1 - efficiency) * 20 + np.random.normal(0, 1)
            pump_vibration = (1 - efficiency) * 2 + np.random.normal(0, 0.1)
            
            # RUL
            if efficiency > failure_threshold:
                rul_hours = (efficiency - failure_threshold) / degradation_rate
                rul_hours = min(rul_hours, 24 * 60)  # Cap at 60 days
            else:
                rul_hours = 0
            
            all_data.append({
                'asset_id': asset_id,
                'timestamp': timestamp,
                'time_days': i,
                'efficiency': efficiency,
                'failure': efficiency < failure_threshold,
                'flow_rate_lpm': flow_rate,
                'coolant_temp_c': coolant_temp,
                'pump_vibration_mm_s': pump_vibration,
                'rul_hours': rul_hours
            })
    
    df = pd.DataFrame(all_data)
    return df

def plot_semiconductor_data(df, asset_id, save_path='assets/semiconductor_case_study.png'):
    """
    Visualize semiconductor degradation data.
    """
    asset_data = df[df['asset_id'] == asset_id]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Health Index
    ax1 = axes[0, 0]
    ax1.plot(asset_data['timestamp'], asset_data['health_index'], 'b-', linewidth=2)
    ax1.axhline(y=0.3, color='red', linestyle='--', label='Failure Threshold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Health Index')
    ax1.set_title(f'Asset {asset_id}: Health Degradation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sensor Trends
    ax2 = axes[0, 1]
    sensor_cols = ['pressure_pa', 'temperature_c', 'motor_current_a', 'vibration_mm_s']
    for col in sensor_cols[:3]:  # Plot first 3 for clarity
        ax2.plot(asset_data['timestamp'], asset_data[col], label=col, alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Sensor Value')
    ax2.set_title('Sensor Trends')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. RUL
    ax3 = axes[1, 0]
    ax3.plot(asset_data['timestamp'], asset_data['rul_hours'], 'g-', linewidth=2)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('RUL (hours)')
    ax3.set_title('Remaining Useful Life')
    ax3.grid(True, alpha=0.3)
    
    # 4. Failure Probability
    ax4 = axes[1, 1]
    # Calculate rolling failure probability
    window = 30
    failure_prob = asset_data['failure'].rolling(window=window).mean() * 100
    ax4.plot(asset_data['timestamp'][window:], failure_prob[window:], 'r-', linewidth=2)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Failure Probability (%)')
    ax4.set_title(f'Rolling Failure Probability (Window: {window} days)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("=" * 60)
    print("SEMICONDUCTOR CASE STUDY SIMULATION")
    print("=" * 60)
    
    print("\n1. Simulating Vacuum Pump Data...")
    pump_data = simulate_vacuum_pump(n_assets=20, n_days=365)
    print(f"   Generated {len(pump_data)} data points")
    print(f"   Features: {list(pump_data.columns)}")
    
    print("\n2. Simulating Cooling System Data...")
    cooling_data = simulate_cooling_system(n_assets=10, n_days=180)
    print(f"   Generated {len(cooling_data)} data points")
    print(f"   Features: {list(cooling_data.columns)}")
    
    print("\n3. Generating visualizations...")
    plot_semiconductor_data(pump_data, asset_id=1)
    
    # Save data
    pump_data.to_csv('data/processed/semiconductor_vacuum_pump.csv', index=False)
    cooling_data.to_csv('data/processed/semiconductor_cooling_system.csv', index=False)
    print("\n   Data saved to data/processed/")
    
    print("\n" + "=" * 60)
    print("SEMICONDUCTOR CASE STUDY COMPLETE")
    print("=" * 60)
