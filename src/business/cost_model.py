import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.rul_predictor import train_xgboost_model, evaluate_model, predict_rul_with_confidence
from simulation.cmapss_loader import load_cmapss, add_rul_labels, drop_constant_sensors
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class BusinessImpactAnalyzer:
    """
    Business Impact Layer - translates technical predictions into financial decisions.
    """
    
    def __init__(self, downtime_cost_per_hour=5000, part_cost=2000, maintenance_cost=1000):
        """
        Initialize cost model with industry-standard rates.
        
        Parameters:
        - downtime_cost_per_hour: Cost of unplanned downtime (£/hour)
        - part_cost: Cost of replacement part (£)
        - maintenance_cost: Cost of scheduled maintenance (£)
        """
        self.downtime_cost_per_hour = downtime_cost_per_hour
        self.part_cost = part_cost
        self.maintenance_cost = maintenance_cost
        
        # Convert cycles to operating hours (assume 1 cycle = 1 hour for this model)
        self.hours_per_cycle = 1.0
        
    def calculate_downtime_cost(self, downtime_hours):
        """
        Calculate cost of unplanned downtime.
        """
        return downtime_hours * self.downtime_cost_per_hour
    
    def calculate_maintenance_cost(self, is_scheduled=True):
        """
        Calculate maintenance cost.
        Scheduled maintenance is cheaper than unscheduled.
        """
        if is_scheduled:
            return self.maintenance_cost
        else:
            # Unscheduled maintenance is more expensive (overtime, expedited parts)
            return self.maintenance_cost * 2.5
    
    def calculate_total_cost(self, rul_hours, maintenance_hours, is_scheduled=True):
        """
        Calculate total cost for a maintenance decision.
        
        Parameters:
        - rul_hours: Predicted RUL in hours
        - maintenance_hours: When maintenance is performed (hours from now)
        - is_scheduled: Whether maintenance is scheduled or unscheduled
        """
        # If maintenance is performed before failure
        if maintenance_hours < rul_hours:
            downtime_hours = 0  # No unplanned downtime
            part_replacement = self.part_cost
            maintenance_cost = self.calculate_maintenance_cost(is_scheduled)
            failure_avoided = True
        else:
            # Failure occurs before maintenance
            downtime_hours = rul_hours * 0.5  # Assume 50% of RUL as downtime (simplified)
            part_replacement = self.part_cost
            maintenance_cost = self.calculate_maintenance_cost(False)
            failure_avoided = False
        
        downtime_cost = self.calculate_downtime_cost(downtime_hours)
        total_cost = downtime_cost + part_replacement + maintenance_cost
        
        return {
            'total_cost': total_cost,
            'downtime_cost': downtime_cost,
            'part_cost': part_replacement,
            'maintenance_cost': maintenance_cost,
            'failure_avoided': failure_avoided,
            'downtime_hours': downtime_hours
        }
    
    def find_optimal_maintenance_window(self, rul_prediction, rul_confidence, 
                                       min_hours=0, max_hours=None):
        """
        Find optimal maintenance window that minimizes total cost.
        
        Parameters:
        - rul_prediction: Predicted RUL (hours)
        - rul_confidence: Standard deviation of prediction
        - min_hours: Minimum hours to wait
        - max_hours: Maximum hours to wait (defaults to predicted RUL)
        """
        if max_hours is None:
            max_hours = int(rul_prediction)
        
        # Evaluate costs at different maintenance times
        maintenance_times = np.arange(min_hours, max_hours + 1, 1)
        costs = []
        
        for t in maintenance_times:
            # Scheduled if we perform maintenance before failure
            is_scheduled = t < rul_prediction
            cost_result = self.calculate_total_cost(rul_prediction, t, is_scheduled)
            costs.append(cost_result['total_cost'])
        
        # Find optimal time
        optimal_idx = np.argmin(costs)
        optimal_time = maintenance_times[optimal_idx]
        optimal_cost = costs[optimal_idx]
        
        # Calculate cost avoidance (compared to reactive maintenance)
        reactive_cost = self.calculate_total_cost(rul_prediction, rul_prediction, False)['total_cost']
        cost_avoidance = reactive_cost - optimal_cost
        
        return {
            'optimal_maintenance_time': optimal_time,
            'optimal_cost': optimal_cost,
            'cost_avoidance': cost_avoidance,
            'reactive_cost': reactive_cost,
            'maintenance_times': maintenance_times,
            'costs': costs
        }
    
    def assess_fleet_risk(self, predictions_df, threshold_hours=50):
        """
        Assess fleet-wide risk based on RUL predictions.
        
        Parameters:
        - predictions_df: DataFrame with 'unit_number', 'rul_prediction', 'rul_std'
        - threshold_hours: Hours below which an asset is considered 'high risk'
        """
        # Identify high-risk assets
        high_risk = predictions_df[predictions_df['rul_prediction'] < threshold_hours]
        medium_risk = predictions_df[(predictions_df['rul_prediction'] >= threshold_hours) & 
                                     (predictions_df['rul_prediction'] < threshold_hours * 2)]
        low_risk = predictions_df[predictions_df['rul_prediction'] >= threshold_hours * 2]
        
        # Calculate total risk cost
        total_risk_cost = 0
        for _, row in predictions_df.iterrows():
            if row['rul_prediction'] < threshold_hours:
                # High risk - likely failure soon
                failure_probability = 1 - (row['rul_prediction'] / threshold_hours)
                expected_cost = failure_probability * self.downtime_cost_per_hour * 24  # 1 day downtime
                total_risk_cost += expected_cost
        
        return {
            'high_risk_count': len(high_risk),
            'medium_risk_count': len(medium_risk),
            'low_risk_count': len(low_risk),
            'high_risk_engines': high_risk['unit_number'].tolist(),
            'total_risk_cost': total_risk_cost,
            'recommendation': f"Prioritize inspection of {len(high_risk)} high-risk assets (RUL < {threshold_hours} hours)"
        }
    
    def generate_executive_summary(self, asset_id, rul_prediction, rul_std, optimal_maintenance):
        """
        Generate executive-friendly summary for an asset.
        """
        summary = {
            'asset_id': asset_id,
            'current_rul_prediction': rul_prediction,
            'rul_confidence_interval': f"±{rul_std * 1.28:.1f} hours (80% CI)",
            'optimal_maintenance_in_hours': optimal_maintenance['optimal_maintenance_time'],
            'optimal_maintenance_cost': f"£{optimal_maintenance['optimal_cost']:,.0f}",
            'reactive_maintenance_cost': f"£{optimal_maintenance['reactive_cost']:,.0f}",
            'cost_avoidance': f"£{optimal_maintenance['cost_avoidance']:,.0f}",
            'recommended_action': f"Schedule maintenance at {optimal_maintenance['optimal_maintenance_time']} hours to avoid £{optimal_maintenance['cost_avoidance']:,.0f} in costs"
        }
        return summary

def demo_business_impact():
    """
    Demonstrate the business impact layer with sample data.
    """
    print("=" * 60)
    print("BUSINESS IMPACT ANALYSIS")
    print("=" * 60)
    
    # Load C-MAPSS data
    print("\n1. Loading data and predicting RUL...")
    df = load_cmapss('FD001', 'train')
    df = add_rul_labels(df)
    df = drop_constant_sensors(df)
    
    # Prepare data for RUL prediction
    from ml.rul_predictor import prepare_rul_data
    X, y, feature_cols = prepare_rul_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    xgb_model = train_xgboost_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluate
    results = evaluate_model(xgb_model, X_test_scaled, y_test, 'XGBoost')
    
    print(f"   Model performance: MAE={results['MAE']:.2f} cycles, R²={results['R2']:.3f}")
    
    # Create business impact analyzer
    print("\n2. Initializing business impact model...")
    analyzer = BusinessImpactAnalyzer(
        downtime_cost_per_hour=5000,   # £5,000 per hour downtime
        part_cost=2000,                # £2,000 replacement part
        maintenance_cost=1000          # £1,000 scheduled maintenance
    )
    
    # Analyze a specific asset (Engine 1)
    print("\n3. Analyzing specific asset (Engine 1):")
    engine_id = 1
    engine_data = df[df['unit_number'] == engine_id]
    X_engine, _, _ = prepare_rul_data(engine_data)
    X_engine_scaled = scaler.transform(X_engine)
    
    # Predict RUL with confidence
    rul_prediction = xgb_model.predict(X_engine_scaled)[-1]  # Most recent prediction
    rul_std = np.std([xgb_model.predict(X_engine_scaled) for _ in range(100)], axis=0)[-1]
    
    print(f"   Current RUL prediction: {rul_prediction:.1f} cycles")
    print(f"   Uncertainty (std dev): ±{rul_std:.1f} cycles")
    
    # Find optimal maintenance window
    optimal = analyzer.find_optimal_maintenance_window(rul_prediction, rul_std)
    
    print(f"\n   Optimal Maintenance Strategy:")
    print(f"   - Perform maintenance at: {optimal['optimal_maintenance_time']:.0f} cycles")
    print(f"   - Expected cost: £{optimal['optimal_cost']:,.0f}")
    print(f"   - Reactive maintenance cost: £{optimal['reactive_cost']:,.0f}")
    print(f"   - Cost avoidance: £{optimal['cost_avoidance']:,.0f}")
    
    # Generate executive summary
    summary = analyzer.generate_executive_summary(engine_id, rul_prediction, rul_std, optimal)
    print(f"\n   Executive Summary:")
    print(f"   - Asset: {summary['asset_id']}")
    print(f"   - RUL: {summary['current_rul_prediction']:.1f} hours")
    print(f"   - Optimal Maintenance: {summary['optimal_maintenance_in_hours']:.0f} hours")
    print(f"   - Cost Avoidance: {summary['cost_avoidance']}")
    print(f"   - Action: {summary['recommended_action']}")
    
    # Fleet-wide risk assessment
    print("\n4. Fleet-wide risk assessment:")
    # Get predictions for all engines (use last cycle of each)
    predictions_df = []
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        X_unit, _, _ = prepare_rul_data(unit_data)
        X_unit_scaled = scaler.transform(X_unit)
        
        # Get prediction for last cycle
        rul_pred = xgb_model.predict(X_unit_scaled)[-1]
        # Estimate uncertainty from bootstrap
        boot_preds = np.array([xgb_model.predict(X_unit_scaled) for _ in range(20)])
        rul_std = np.std(boot_preds, axis=0)[-1]
        
        predictions_df.append({
            'unit_number': unit,
            'rul_prediction': rul_pred,
            'rul_std': rul_std
        })
    
    predictions_df = pd.DataFrame(predictions_df)
    
    risk_assessment = analyzer.assess_fleet_risk(predictions_df, threshold_hours=50)
    
    print(f"   High-risk engines (RUL < 50 cycles): {risk_assessment['high_risk_count']}")
    print(f"   Medium-risk engines: {risk_assessment['medium_risk_count']}")
    print(f"   Low-risk engines: {risk_assessment['low_risk_count']}")
    print(f"   Estimated total risk cost: £{risk_assessment['total_risk_cost']:,.0f}")
    print(f"   Recommendation: {risk_assessment['recommendation']}")
    
    if risk_assessment['high_risk_count'] > 0:
        print(f"   High-risk engine IDs: {risk_assessment['high_risk_engines'][:5]}")
    
    # Plot cost optimization curve
    print("\n5. Generating cost optimization visualization...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    maintenance_times = optimal['maintenance_times']
    costs = optimal['costs']
    
    ax.plot(maintenance_times, costs, 'b-', linewidth=2)
    ax.axvline(x=optimal['optimal_maintenance_time'], color='red', linestyle='--', 
               label=f"Optimal: {optimal['optimal_maintenance_time']:.0f} hours")
    ax.axhline(y=optimal['reactive_cost'], color='orange', linestyle='--', 
               label=f"Reactive: £{optimal['reactive_cost']:,.0f}")
    ax.set_xlabel('Maintenance Time (hours from now)')
    ax.set_ylabel('Total Cost (£)')
    ax.set_title('Maintenance Cost Optimization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/cost_optimization.png', dpi=300, bbox_inches='tight')
    print("   Cost optimization plot saved to assets/cost_optimization.png")
    plt.show()
    
    # Save fleet risk assessment
    predictions_df.to_csv('data/processed/fleet_risk_assessment.csv', index=False)
    print("\n   Fleet risk assessment saved to data/processed/fleet_risk_assessment.csv")
    
    print("\n" + "=" * 60)
    print("BUSINESS IMPACT ANALYSIS COMPLETE")
    print("=" * 60)
    
    return analyzer, predictions_df

if __name__ == "__main__":
    analyzer, predictions_df = demo_business_impact()
