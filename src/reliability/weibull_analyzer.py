import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import WeibullFitter
from lifelines.plotting import qq_plot
import sys
import os

# Add parent directory to path so we can import the loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.cmapss_loader import load_cmapss, add_rul_labels, drop_constant_sensors

def extract_failure_times(df):
    """
    Extract failure times from C-MAPSS data.
    For each engine, find the maximum cycle (when it failed).
    """
    failure_times = df.groupby('unit_number')['time_in_cycles'].max().values
    return failure_times

def fit_weibull(failure_times):
    """
    Fit Weibull distribution to failure times using lifelines.
    Returns the fitted model and parameters.
    """
    # WeibullFitter expects event data (all events are observed since we have run-to-failure)
    wf = WeibullFitter()
    
    # Create a DataFrame with failure times and event indicator (all observed = 1)
    df_weibull = pd.DataFrame({
        'failure_time': failure_times,
        'event': np.ones(len(failure_times))
    })
    
    # Fit the model
    wf.fit(df_weibull['failure_time'], event_observed=df_weibull['event'])
    
    return wf

def calculate_weibull_metrics(wf, failure_times):
    """
    Calculate key reliability metrics from Weibull fit.
    """
    # Extract parameters
    rho = wf.rho_  # Shape parameter (beta)
    lambda_ = wf.lambda_  # Scale parameter (eta)
    
    # Calculate MTBF (mean time between failures) = eta * Gamma(1 + 1/beta)
    from scipy.special import gamma
    mtbf = lambda_ * gamma(1 + 1/rho)
    
    # Calculate characteristic life (time to 63.2% failure)
    char_life = lambda_  # In Weibull, eta is the characteristic life
    
    # Calculate B10 life (time to 10% failure)
    b10_life = lambda_ * (-np.log(0.9)) ** (1/rho)
    
    # Calculate B50 life (median time to failure)
    b50_life = lambda_ * (-np.log(0.5)) ** (1/rho)
    
    return {
        'shape_parameter_beta': rho,
        'scale_parameter_eta': lambda_,
        'mtbf': mtbf,
        'characteristic_life': char_life,
        'b10_life': b10_life,
        'b50_life': b50_life
    }

def plot_weibull_analysis(wf, failure_times, save_path='assets/weibull_analysis.png'):
    """
    Create Weibull probability plot and survival curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Weibull Probability Plot
    wf.plot(ax=axes[0])
    axes[0].set_title('Weibull Probability Plot', fontsize=14)
    axes[0].set_xlabel('Time (cycles)')
    axes[0].set_ylabel('Survival Probability')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Survival Function
    survival_times = np.linspace(0, max(failure_times), 100)
    survival_prob = wf.survival_function_at_times(survival_times).values
    
    axes[1].plot(survival_times, survival_prob, 'b-', linewidth=2)
    axes[1].scatter(failure_times, np.sort(wf.survival_function_at_times(failure_times).values), 
                   alpha=0.3, color='red', s=30)
    axes[1].set_title('Survival Function (Kaplan-Meier style)', fontsize=14)
    axes[1].set_xlabel('Time (cycles)')
    axes[1].set_ylabel('Survival Probability')
    axes[1].grid(True, alpha=0.3)
    
    # Add annotation for B10 and B50
    metrics = calculate_weibull_metrics(wf, failure_times)
    ax = axes[1]
    ax.axvline(metrics['b10_life'], color='green', linestyle='--', alpha=0.7, label=f"B10 Life: {metrics['b10_life']:.1f}")
    ax.axvline(metrics['b50_life'], color='orange', linestyle='--', alpha=0.7, label=f"B50 Life: {metrics['b50_life']:.1f}")
    ax.legend()
    
    plt.tight_layout()
    
    # Ensure assets directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    return fig

def calculate_failure_probability(wf, time_horizon):
    """
    Calculate failure probability at a given time horizon.
    P(failure <= t) = 1 - exp(-(t/eta)^beta)
    """
    eta = wf.lambda_
    beta = wf.rho_
    failure_prob = 1 - np.exp(-((time_horizon / eta) ** beta))
    return failure_prob

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("WEIBULL ANALYSIS FOR TURBOFAN ENGINE FAILURE TIMES")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading C-MAPSS FD001 data...")
    df = load_cmapss('FD001', 'train')
    df = add_rul_labels(df)
    df = drop_constant_sensors(df)
    
    # Extract failure times
    print("2. Extracting failure times for each engine...")
    failure_times = extract_failure_times(df)
    print(f"   Found {len(failure_times)} engines with failure times")
    print(f"   Failure time range: {min(failure_times)} to {max(failure_times)} cycles")
    print(f"   Mean failure time: {np.mean(failure_times):.1f} cycles")
    print(f"   Std deviation: {np.std(failure_times):.1f} cycles")
    
    # Fit Weibull
    print("\n3. Fitting Weibull distribution...")
    wf = fit_weibull(failure_times)
    
    # Calculate metrics
    print("\n4. Reliability Metrics:")
    metrics = calculate_weibull_metrics(wf, failure_times)
    print(f"   Shape parameter (β): {metrics['shape_parameter_beta']:.3f}")
    print(f"   Scale parameter (η): {metrics['scale_parameter_eta']:.1f} cycles")
    print(f"   MTBF: {metrics['mtbf']:.1f} cycles")
    print(f"   Characteristic life (63.2% failure): {metrics['characteristic_life']:.1f} cycles")
    print(f"   B10 life (10% failure): {metrics['b10_life']:.1f} cycles")
    print(f"   B50 life (50% failure): {metrics['b50_life']:.1f} cycles")
    
    # Calculate failure probabilities at key horizons
    print("\n5. Failure Probabilities at Different Horizons:")
    for horizon in [50, 100, 150, 200, 250]:
        prob = calculate_failure_probability(wf, horizon)
        print(f"   P(failure <= {horizon} cycles): {prob*100:.2f}%")
    
    # Create plots
    print("\n6. Generating Weibull plots...")
    plot_weibull_analysis(wf, failure_times)
    
    print("\n" + "=" * 60)
    print("WEIBULL ANALYSIS COMPLETE")
    print("=" * 60)
