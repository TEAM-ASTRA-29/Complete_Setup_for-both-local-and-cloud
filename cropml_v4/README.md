# CropML Research Studio v4.0

A dynamic, research-grade ML comparison platform supporting Regression and
Classification tasks with smart model recommendation, preprocessing pipeline,
PDF report export, and cloud vs local deployment simulation.

---

## Project Structure

```
cropml_v4/
│
├── app.py                        ← Streamlit entry point (run this)
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── backend/
│   ├── __init__.py
│   ├── models.py                 ← Model registry, builder, trainer, recommender,
│   │                               cloud vs local simulator
│   ├── preprocessor.py           ← DataProfiler + PreprocessingPipeline
│   ├── report.py                 ← PDF report generator (ReportLab + Kaleido)
│   └── session.py                ← Session save/load/delete helpers
│
├── frontend/
│   ├── __init__.py
│   ├── components.py             ← CSS, KPI cards, badges, all Plotly chart builders,
│   │                               metric comparison table
│   └── hyperparams.py            ← Per-model hyperparameter slider widgets
│
├── utils/
│   └── __init__.py
│
└── saved_sessions/               ← Auto-created; JSON files per training session
    └── <session_id>.json
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open browser at http://localhost:8501
```

---

## Features

### 1. Preprocessing Pipeline (raw OR pre-processed CSV)
- Auto-detects whether uploaded data is already clean
- If pre-processed: skips encoding/imputation with one checkbox
- If raw: full pipeline with label/one-hot encoding, mean/median/mode/drop null handling,
  duplicate row removal, low-variance feature dropping, StandardScaler/MinMaxScaler
- Preprocessing log shown in UI and included in PDF report

### 2. Smart Model Recommendation (3 modes)
- **Auto (dataset-aware)**: scores every model based on dataset size, feature count,
  null presence, and categorical ratio — shows top 5 with scores and reasons
- **Rule-based**: user sets priorities (prefer speed / accuracy / interpretability /
  small model / max training time) and the system adjusts scores accordingly
- **Manual**: user picks any models freely

### 3. Regression Models (13)
| Model | Key | Notes |
|---|---|---|
| Random Forest | rf | Ensemble bagging |
| XGBoost | xgb | Gradient boosting, fast |
| Gradient Boosting | gbm | sklearn GBM |
| Extra Trees | et | Randomised bagging |
| AdaBoost | ada | Adaptive boosting |
| Decision Tree | dt | Single tree, interpretable |
| Ridge Regression | ridge | L2 regularised linear |
| Lasso Regression | lasso | L1 regularised, feature selection |
| ElasticNet | enet | L1+L2 combined |
| Linear Regression | lr | OLS, no hyperparams |
| SVR (SVM) | svr | Kernel SVM regression |
| K-Nearest Neighbors | knn | Instance-based |
| MLP Neural Network | mlp | Feedforward NN |

### 4. Classification Models (10)
| Model | Key |
|---|---|
| Random Forest | rf |
| XGBoost | xgb |
| Gradient Boosting | gbm |
| Extra Trees | et |
| AdaBoost | ada |
| Decision Tree | dt |
| Logistic Regression | logreg |
| SVC (SVM) | svc |
| K-Nearest Neighbors | knn |
| MLP Neural Network | mlp |

### 5. Pages
| Page | Description |
|---|---|
| 🔬 Train | Upload CSV, configure preprocessing, get recommendations, tune hyperparams, train |
| 📊 Analysis | Full visual analysis of active session — charts, inferences, PDF export |
| 📐 Compare | Detailed metric comparison table with definitions + cloud vs local simulation |
| ⚖️ Cross-Session | Compare multiple training sessions side by side |
| 📁 History | Browse, activate, or delete all saved sessions |

### 6. Visualisations
- R²/Accuracy bar chart
- RMSE/MAE side-by-side bars
- F1/Precision/Recall grouped bars
- Actual vs Predicted scatter
- Residual scatter + histogram overlay
- Feature importance (gradient colour bar, top 20)
- Computational performance (runtime, memory, model size)
- Radar profile (6-dimensional normalised)
- Confusion matrix (heatmap)
- Cloud vs Local bar comparison
- Cross-session heatmap

### 7. Metric Comparison Table
Full HTML table with:
- All metrics for all trained models side by side
- Green highlight = best value per metric
- Orange highlight = worst value per metric
- Metric definitions and usage guidance

### 8. Cloud vs Local Simulation
Supported AWS instances:
- t3.micro, t3.medium, t3.large
- c5.xlarge, c5.4xlarge
- m5.4xlarge
- p3.2xlarge (GPU)

Shows: estimated runtime, speedup factor, memory usage, cost per training run.

### 9. PDF Report
Includes: session metadata, full metric table, metric definitions, hyperparameter
tables per model, primary metric chart, runtime chart, feature importance charts
(up to 4 models), cloud vs local table if simulated.

### 10. Session Persistence
Sessions saved as JSON in `saved_sessions/`. Auto-loaded on app restart.
Each session stores: all metrics, predictions, feature importances, hyperparams,
preprocessing log, y_test values for residual plots.

---

## Metrics Reference

### Regression
| Metric | Range | Better |
|---|---|---|
| R² | [0,1] (can be negative) | Higher |
| RMSE | [0,∞) | Lower |
| MAE | [0,∞) | Lower |
| MSE | [0,∞) | Lower |

### Classification
| Metric | Range | Better |
|---|---|---|
| Accuracy | [0,1] | Higher |
| F1 | [0,1] | Higher |
| Precision | [0,1] | Higher |
| Recall | [0,1] | Higher |
| AUC | [0,1] | Higher |
