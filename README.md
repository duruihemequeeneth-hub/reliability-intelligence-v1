# Reliability Intelligence Platform V1

**A prototype demonstrating system simulation, machine learning, reliability analytics, RUL prediction, and executive decision support for aerospace and semiconductor applications.**

---

## What This Platform Does

| Module | Description | Technology |
|--------|-------------|------------|
| **Simulation Engine** | C-MAPSS turbofan degradation data | Python, NumPy |
| **Reliability Analytics** | Weibull analysis, MTBF, B10/B50 life | Lifelines, SciPy |
| **Health Monitoring** | Anomaly detection, degradation trends | Isolation Forest, Scikit-learn |
| **RUL Prediction** | Remaining useful life with confidence intervals | XGBoost, Bootstrap |
| **Business Impact** | Cost optimization, risk assessment | Custom cost model |
| **Executive Dashboard** | Interactive decision support | Streamlit, Plotly |

---

## Key Results

- **Weibull Shape Parameter (β):**
- **MTBF:**
- **B10 Life:**
- **RUL Prediction MAE:**
- **R² Score:**
- **Cost Avoidance:**
- **Fleet Risk:**

---

##  Architecture
┌─────────────────────────────────────────────────────────────┐
│ Streamlit Executive Dashboard │
│ - Asset health heatmap │
│ - RUL with confidence bands │
│ - Cost optimizer widget │
│ - Actionable recommendations │
└─────────────────────────────────────────────────────────────┘
▲
┌─────────────────────────────────────────────────────────────┐
│ Business Impact Layer │
│ - Cost optimization │
│ - Risk assessment │
│ - Executive summaries │
└─────────────────────────────────────────────────────────────┘
▲
┌─────────────────────────────────────────────────────────────┐
│ ML & Reliability Layer │
│ - XGBoost RUL prediction │
│ - Isolation Forest anomaly detection │
│ - Weibull survival analysis │
└─────────────────────────────────────────────────────────────┘
▲
┌─────────────────────────────────────────────────────────────┐
│ Data Layer │
│ - NASA C-MAPSS turbofan data (FD001) │
│ - Synthetic semiconductor data (planned) │
└─────────────────────────────────────────────────────────────┘
