import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import joblib
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.cmapss_loader import load_cmapss, add_rul_labels, drop_constant_sensors
from src.ml.rul_predictor import prepare_rul_data
from src.business.cost_model import BusinessImpactAnalyzer
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Reliability Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Reliability Intelligence Platform V1")
st.markdown("*Executive Decision Support for Asset Reliability*")

# Load data and models
@st.cache_data
def load_data():
    """Load and cache C-MAPSS data."""
    df = load_cmapss('FD001', 'train')
    df = add_rul_labels(df)
    df = drop_constant_sensors(df)
    return df

@st.cache_resource
def load_models():
    """Load trained models."""
    try:
        xgb_model = joblib.load('models/xgboost_rul_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return xgb_model, scaler
    except:
        return None, None

# Load everything
df = load_data()
xgb_model, scaler = load_models()

# Sidebar
st.sidebar.header(" Navigation & Controls")

# Asset selection
engine_ids = sorted(df['unit_number'].unique())
selected_engine = st.sidebar.selectbox(
    "Select Asset (Engine)",
    engine_ids,
    index=0,
    help="Choose an engine to analyze"
)

# Cost parameters
st.sidebar.subheader(" Cost Parameters")
downtime_cost = st.sidebar.number_input(
    "Downtime Cost (£/hour)",
    min_value=1000,
    max_value=50000,
    value=5000,
    step=500,
    help="Cost of unplanned downtime per hour"
)

part_cost = st.sidebar.number_input(
    "Replacement Part Cost (£)",
    min_value=500,
    max_value=10000,
    value=2000,
    step=500,
    help="Cost of replacement part"
)

maintenance_cost = st.sidebar.number_input(
    "Scheduled Maintenance Cost (£)",
    min_value=500,
    max_value=5000,
    value=1000,
    step=100,
    help="Cost of scheduled maintenance"
)

# Thresholds
st.sidebar.subheader("Risk Thresholds")
high_risk_threshold = st.sidebar.slider(
    "High Risk Threshold (cycles)",
    min_value=20,
    max_value=100,
    value=50,
    step=5,
    help="Assets with RUL below this are high risk"
)

# Main content
if xgb_model is not None and scaler is not None:
    
    # Get data for selected engine
    engine_data = df[df['unit_number'] == selected_engine]
    X_engine, _, _ = prepare_rul_data(engine_data)
    X_engine_scaled = scaler.transform(X_engine)
    
    # Get predictions for all cycles
    rul_predictions = xgb_model.predict(X_engine_scaled)
    
    # Get latest prediction
    latest_rul = rul_predictions[-1]
    
    # Calculate uncertainty (bootstrap)
    n_bootstrap = 50
    bootstrap_preds = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_engine_scaled), len(X_engine_scaled), replace=True)
        boot_pred = xgb_model.predict(X_engine_scaled[idx])
        bootstrap_preds.append(boot_pred[-1])
    rul_std = np.std(bootstrap_preds)
    rul_ci_lower = np.percentile(bootstrap_preds, 10)
    rul_ci_upper = np.percentile(bootstrap_preds, 90)
    
    # Business impact analyzer
    analyzer = BusinessImpactAnalyzer(
        downtime_cost_per_hour=downtime_cost,
        part_cost=part_cost,
        maintenance_cost=maintenance_cost
    )
    
    # Find optimal maintenance
    optimal = analyzer.find_optimal_maintenance_window(latest_rul, rul_std)
    summary = analyzer.generate_executive_summary(
        selected_engine, latest_rul, rul_std, optimal
    )
    
    # --- TOP ROW: Executive Summary Cards ---
    st.subheader("Executive Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Remaining Useful Life",
            value=f"{latest_rul:.1f} cycles",
            delta=f"±{rul_std:.1f} cycles (80% CI)"
        )
    
    with col2:
        st.metric(
            label="Optimal Maintenance Window",
            value=f"{optimal['optimal_maintenance_time']:.0f} cycles",
            delta="from now"
        )
    
    with col3:
        st.metric(
            label="Cost Avoidance",
            value=f"£{optimal['cost_avoidance']:,.0f}",
            delta=f"vs. reactive £{optimal['reactive_cost']:,.0f}",
            delta_color="normal"
        )
    
    with col4:
        # Risk status
        if latest_rul < high_risk_threshold:
            status = "HIGH RISK"
            status_color = "red"
        elif latest_rul < high_risk_threshold * 2:
            status = "MEDIUM RISK"
            status_color = "orange"
        else:
            status = "LOW RISK"
            status_color = "green"
        
        st.metric(
            label="Risk Status",
            value=status,
            delta=f"Threshold: {high_risk_threshold} cycles"
        )
    
    # --- MIDDLE ROW: Charts ---
    st.subheader("Asset Health & Degradation")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "RUL Prediction", 
        "Cost Optimization", 
        "Sensor Trends", 
        "Fleet Risk",
        "What-If Analysis" 
    ])
    
    with tab1:
        # RUL prediction chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("RUL Prediction Over Time", "Prediction Distribution"),
            vertical_spacing=0.15
        )
        
        # Plot RUL over time
        time_cycles = engine_data['time_in_cycles'].values
        
        # Actual RUL (if available)
        if 'rul' in engine_data.columns:
            actual_rul = engine_data['rul'].values
            
            fig.add_trace(
                go.Scatter(
                    x=time_cycles,
                    y=actual_rul,
                    mode='lines',
                    name='Actual RUL',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Predicted RUL
        fig.add_trace(
            go.Scatter(
                x=time_cycles,
                y=rul_predictions,
                mode='lines',
                name='Predicted RUL',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Confidence interval
        # Simplified: use ±1.28*std for 80% CI
        fig.add_trace(
            go.Scatter(
                x=time_cycles,
                y=rul_predictions + 1.28 * rul_std,
                mode='lines',
                name='Upper CI (80%)',
                line=dict(color='rgba(255,0,0,0.2)', width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=time_cycles,
                y=rul_predictions - 1.28 * rul_std,
                mode='lines',
                name='Lower CI (80%)',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,0,0,0.2)', width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add vertical line for current time
        fig.add_vline(
            x=time_cycles[-1],
            line_dash="dash",
            line_color="gray",
            annotation_text="Now",
            row=1, col=1
        )
        
        fig.update_xaxes(title_text="Time (cycles)", row=1, col=1)
        fig.update_yaxes(title_text="RUL (cycles)", row=1, col=1)
        
        # Distribution of predictions (bootstrap)
        fig.add_trace(
            go.Histogram(
                x=bootstrap_preds,
                nbinsx=20,
                name='Bootstrap Predictions',
                marker_color='blue',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.add_vline(
            x=latest_rul,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {latest_rul:.1f}",
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="RUL (cycles)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(
            height=600,
            showlegend=True,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Cost optimization chart
        maintenance_times = optimal['maintenance_times']
        costs = optimal['costs']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=maintenance_times,
            y=costs,
            mode='lines',
            name='Total Cost',
            line=dict(color='blue', width=3)
        ))
        
        # Optimal point
        fig.add_trace(go.Scatter(
            x=[optimal['optimal_maintenance_time']],
            y=[optimal['optimal_cost']],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name=f"Optimal: {optimal['optimal_maintenance_time']:.0f} hours"
        ))
        
        # Reactive cost line
        fig.add_hline(
            y=optimal['reactive_cost'],
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Reactive: £{optimal['reactive_cost']:,.0f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title="Maintenance Cost Optimization",
            xaxis_title="Maintenance Time (cycles from now)",
            yaxis_title="Total Cost (£)",
            height=500,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost breakdown
        st.subheader("Cost Breakdown")
        
        col1, col2, col3 = st.columns(3)
        
        reactive_result = analyzer.calculate_total_cost(latest_rul, latest_rul, False)
        optimal_result = analyzer.calculate_total_cost(latest_rul, optimal['optimal_maintenance_time'], True)
        
        with col1:
            st.metric(
                "Reactive Maintenance",
                f"£{reactive_result['total_cost']:,.0f}",
                "Failure occurs"
            )
            st.caption(f"• Downtime: £{reactive_result['downtime_cost']:,.0f}")
            st.caption(f"• Part: £{reactive_result['part_cost']:,.0f}")
            st.caption(f"• Labor: £{reactive_result['maintenance_cost']:,.0f}")
        
        with col2:
            st.metric(
                "Optimal Maintenance",
                f"£{optimal_result['total_cost']:,.0f}",
                f"Savings: £{optimal['cost_avoidance']:,.0f}",
                delta_color="normal"
            )
            st.caption(f"• Downtime: £{optimal_result['downtime_cost']:,.0f}")
            st.caption(f"• Part: £{optimal_result['part_cost']:,.0f}")
            st.caption(f"• Labor: £{optimal_result['maintenance_cost']:,.0f}")
        
        with col3:
            st.metric(
                "Cost Avoidance",
                f"£{optimal['cost_avoidance']:,.0f}",
                f"{optimal['cost_avoidance']/optimal['reactive_cost']*100:.0f}% savings",
                delta_color="normal"
            )
            st.caption(f"• Per asset savings")
            st.caption(f"• Across fleet: £{optimal['cost_avoidance'] * 100:,.0f}")
    
    with tab3:
        # Sensor trends
        st.subheader("Key Sensor Trends")
        
        # Select sensors to display
        sensor_cols = [col for col in engine_data.columns if col.startswith('sensor_')]
        selected_sensors = st.multiselect(
            "Select sensors to display",
            sensor_cols,
            default=sensor_cols[:3]
        )
        
        if selected_sensors:
            fig = go.Figure()
            
            for sensor in selected_sensors:
                fig.add_trace(go.Scatter(
                    x=engine_data['time_in_cycles'],
                    y=engine_data[sensor],
                    mode='lines',
                    name=sensor,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Sensor Trends Over Time",
                xaxis_title="Time (cycles)",
                yaxis_title="Sensor Value",
                height=500,
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Fleet risk assessment
        st.subheader("Fleet-wide Risk Assessment")
        
        # Get predictions for all engines
        predictions_df = []
        for unit in engine_ids:
            unit_data = df[df['unit_number'] == unit]
            X_unit, _, _ = prepare_rul_data(unit_data)
            X_unit_scaled = scaler.transform(X_unit)
            
            rul_pred = xgb_model.predict(X_unit_scaled)[-1]
            
            predictions_df.append({
                'unit_number': unit,
                'rul_prediction': rul_pred
            })
        
        predictions_df = pd.DataFrame(predictions_df)
        
        # Risk categories
        high_risk = predictions_df[predictions_df['rul_prediction'] < high_risk_threshold]
        medium_risk = predictions_df[(predictions_df['rul_prediction'] >= high_risk_threshold) & 
                                      (predictions_df['rul_prediction'] < high_risk_threshold * 2)]
        low_risk = predictions_df[predictions_df['rul_prediction'] >= high_risk_threshold * 2]
        
        # Risk summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("High Risk", len(high_risk), f"RUL < {high_risk_threshold} cycles")
        with col2:
            st.metric("Medium Risk", len(medium_risk), f"{high_risk_threshold} - {high_risk_threshold*2} cycles")
        with col3:
            st.metric("Low Risk", len(low_risk), f"RUL > {high_risk_threshold*2} cycles")
        with col4:
            total_risk = analyzer.assess_fleet_risk(predictions_df, high_risk_threshold)['total_risk_cost']
            st.metric("Total Risk Cost", f"£{total_risk:,.0f}")
        
        # Risk distribution chart
        fig = px.histogram(
            predictions_df,
            x='rul_prediction',
            nbins=20,
            title='Fleet RUL Distribution',
            labels={'rul_prediction': 'RUL (cycles)'},
            color_discrete_sequence=['blue']
        )
        
        # Add threshold lines
        fig.add_vline(x=high_risk_threshold, line_dash="dash", line_color="red", 
                      annotation_text="High Risk")
        fig.add_vline(x=high_risk_threshold*2, line_dash="dash", line_color="orange",
                      annotation_text="Medium Risk")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # High-risk list
        if len(high_risk) > 0:
            st.subheader("High-Risk Assets (Immediate Attention)")
            st.dataframe(
                high_risk.sort_values('rul_prediction'),
                use_container_width=True,
                hide_index=True
            )
    
    # --- BOTTOM ROW: Actionable Recommendations ---
    st.subheader("Recommended Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(
            f"**Priority 1: Asset {selected_engine}**\n\n"
            f"• **Action:** {summary['recommended_action']}\n"
            f"• **Timing:** Within {optimal['optimal_maintenance_time']:.0f} cycles\n"
            f"• **Expected Savings:** {summary['cost_avoidance']}\n"
            f"• **Confidence:** 80% CI (±{rul_std * 1.28:.1f} cycles)"
        )
    
    with col2:
        if len(high_risk) > 0:
            st.warning(
                f"**Fleet-wide Priority**\n\n"
                f"• **High-risk assets:** {len(high_risk)} engines need immediate inspection\n"
                f"• **Top 5 IDs:** {high_risk['unit_number'].head(5).tolist()}\n"
                f"• **Total risk exposure:** £{total_risk:,.0f}\n"
                f"• **Recommendation:** Deploy inspection team to high-risk assets"
            )
        else:
            st.success(
                "**Fleet Status: Healthy**\n\n"
                f"• No assets are currently high-risk\n"
                f"• {len(medium_risk)} assets in medium-risk zone\n"
                f"• Continue regular monitoring"
            )

else:
    st.error("Models not found. Please train the RUL predictor first.")
    st.info("Run: `python src/ml/rul_predictor.py` to train models"
    )

    with tab5:
        st.subheader(" What-If Sensitivity Analysis")
        
        st.markdown("""
        **Explore how changes in cost parameters affect optimal maintenance decisions.**
        This sensitivity analysis helps you understand the robustness of recommendations.
        """)
        
        # Only show if we have a valid RUL prediction
        if latest_rul > 0:
            # Sensitivity sliders
            col1, col2 = st.columns(2)
            
            with col1:
                # Maintenance time range
                max_range = int(min(latest_rul * 1.5, 200))
                min_range = 0
                
                if max_range > 1:
                    sensitivity_hours = st.slider(
                        "Maintenance Time Range (cycles)",
                        min_value=min_range,
                        max_value=max_range,
                        value=(min_range, min(max_range, int(latest_rul) if latest_rul > 0 else 20)),
                        step=1,
                        help="Range of maintenance times to evaluate"
                    )
                else:
                    st.warning("RUL is too small for sensitivity analysis. Select a different asset.")
                    sensitivity_hours = (0, 10)
            
            with col2:
                cost_multiplier = st.slider(
                    "Cost Scenario Multiplier",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Scales all costs up/down to test sensitivity"
                )
            
            # Run sensitivity analysis if we have a valid range
            if latest_rul > 0 and sensitivity_hours[1] > sensitivity_hours[0]:
                # Generate cost curves for different scenarios
                maintenance_times = np.arange(
                    max(0, sensitivity_hours[0]),
                    min(int(latest_rul * 1.5), sensitivity_hours[1] + 1),
                    1
                )
                
                if len(maintenance_times) > 1:
                    # --- Base Case ---
                    base_costs = []
                    for t in maintenance_times:
                        result = analyzer.calculate_total_cost(latest_rul, t, t < latest_rul)
                        base_costs.append(result['total_cost'] * cost_multiplier)
                    
                    # --- Optimistic Case (lower costs) ---
                    low_costs = []
                    for t in maintenance_times:
                        analyzer_low = BusinessImpactAnalyzer(
                            downtime_cost_per_hour=downtime_cost * 0.7,
                            part_cost=part_cost * 0.7,
                            maintenance_cost=maintenance_cost * 0.7
                        )
                        result = analyzer_low.calculate_total_cost(latest_rul, t, t < latest_rul)
                        low_costs.append(result['total_cost'])
                    
                    # --- Pessimistic Case (higher costs) ---
                    high_costs = []
                    for t in maintenance_times:
                        analyzer_high = BusinessImpactAnalyzer(
                            downtime_cost_per_hour=downtime_cost * 1.3,
                            part_cost=part_cost * 1.3,
                            maintenance_cost=maintenance_cost * 1.3
                        )
                        result = analyzer_high.calculate_total_cost(latest_rul, t, t < latest_rul)
                        high_costs.append(result['total_cost'])
                    
                    # --- Plot Sensitivity ---
                    fig = go.Figure()
                    
                    # Base case
                    fig.add_trace(go.Scatter(
                        x=maintenance_times,
                        y=base_costs,
                        mode='lines',
                        name='Base Case',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Optimistic case
                    fig.add_trace(go.Scatter(
                        x=maintenance_times,
                        y=low_costs,
                        mode='lines',
                        name='Optimistic (70% costs)',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                    
                    # Pessimistic case
                    fig.add_trace(go.Scatter(
                        x=maintenance_times,
                        y=high_costs,
                        mode='lines',
                        name='Pessimistic (130% costs)',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                    
                    # Find optimal points for base case
                    optimal_idx = np.argmin(base_costs)
                    optimal_time = maintenance_times[optimal_idx]
                    optimal_cost = base_costs[optimal_idx]
                    
                    # Mark optimal point
                    fig.add_trace(go.Scatter(
                        x=[optimal_time],
                        y=[optimal_cost],
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='star'),
                        name=f'Optimal: {optimal_time:.0f} cycles'
                    ))
                    
                    # Add vertical line for optimal
                    fig.add_vline(
                        x=optimal_time,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Optimal: {optimal_time:.0f}",
                        annotation_position="top"
                    )
                    
                    fig.update_layout(
                        title='Sensitivity Analysis: Cost Scenarios',
                        xaxis_title='Maintenance Time (cycles from now)',
                        yaxis_title=f'Total Cost (£)',
                        height=500,
                        hovermode='x',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- Interpretation ---
                    st.subheader(" Interpretation & Recommendations")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Optimal Maintenance Window",
                            f"{optimal_time:.0f} cycles",
                            "from now"
                        )
                    
                    with col2:
                        st.metric(
                            "Cost Range at Optimum",
                            f"£{min(low_costs):,.0f} - £{max(high_costs):,.0f}",
                            f"Base: £{optimal_cost:,.0f}"
                        )
                    
                    with col3:
                        # Calculate stability of optimal window
                        # Find optimal for each scenario
                        opt_low = maintenance_times[np.argmin(low_costs)]
                        opt_high = maintenance_times[np.argmin(high_costs)]
                        stability = abs(opt_high - opt_low)
                        
                        if stability <= 5:
                            stability_status = " High"
                            stability_color = "green"
                        elif stability <= 15:
                            stability_status = " Medium"
                            stability_color = "orange"
                        else:
                            stability_status = " Low"
                            stability_color = "red"
                        
                        st.metric(
                            "Recommendation Stability",
                            stability_status,
                            f"±{stability:.0f} cycles variation"
                        )
                    
                    # Detailed interpretation
                    st.info(f"""
                    **Analysis Summary:**
                    
                    - **Optimal Maintenance Window:** {optimal_time:.0f} cycles from now
                    - **Cost Savings vs. Reactive:** £{optimal['cost_avoidance'] * cost_multiplier:,.0f}
                    - **Stability Assessment:** The optimal window {'is stable across cost scenarios' if stability <= 10 else 'varies with cost assumptions'}
                    
                    **Recommendation:**
                    {' Proceed with scheduled maintenance at the optimal window. The recommendation is robust.' if stability <= 10 else ' Consider conducting maintenance earlier or later depending on your risk tolerance.'}
                    
                    **Next Steps:**
                    - Monitor asset condition leading up to the optimal window
                    - Prepare maintenance resources in advance
                    - Consider a mid-cycle inspection if concerns arise
                    """)
                    
                    # --- Cost Scenario Comparison Table ---
                    st.subheader(" Scenario Comparison")
                    
                    scenario_data = pd.DataFrame({
                        'Scenario': ['Base Case', 'Optimistic (70% costs)', 'Pessimistic (130% costs)'],
                        'Optimal Maintenance': [f"{optimal_time:.0f} cycles", f"{opt_low:.0f} cycles", f"{opt_high:.0f} cycles"],
                        'Total Cost': [
                            f"£{optimal_cost:,.0f}",
                            f"£{min(low_costs):,.0f}",
                            f"£{max(high_costs):,.0f}"
                        ],
                        'Cost Avoidance': [
                            f"£{optimal['cost_avoidance'] * cost_multiplier:,.0f}",
                            f"£{optimal['cost_avoidance'] * 0.7:,.0f}",
                            f"£{optimal['cost_avoidance'] * 1.3:,.0f}"
                        ]
                    })
                    
                    st.table(scenario_data)
                    
                else:
                    st.warning("Adjust sensitivity range to include more maintenance times.")
            else:
                st.warning("RUL is too small for meaningful sensitivity analysis. Select a different asset with RUL > 10 cycles.")
        else:
            st.warning(" This asset has already failed or is at end-of-life (RUL ≤ 0 cycles).")
            st.info("Select a different asset from the sidebar to analyze.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("🔧 Reliability Intelligence Platform V1")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
st.sidebar.caption("Built with Python, XGBoost, Streamlit")
