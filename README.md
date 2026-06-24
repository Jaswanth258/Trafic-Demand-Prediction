# Traffic Demand Prediction

Predicting urban traffic demand (continuous value between `[0, 1]`) at specific grid locations (`geohash`) and time intervals, optimized using the $R^2$ evaluation metric.

---

## 🚀 Project Overview & Key Challenges

This project addresses a challenging traffic forecasting problem defined by a distribution mismatch between the training and testing sets:
*   **Train Set**: Day 48 (covers all 24 hours) and Day 49 (covers only hours 0 to 2, i.e., 12:00 AM to 2:00 AM).
*   **Test Set**: Day 49 (covers hours 2:15 to 13:45).
*   **The GBDT Extrapolation Trap**: Because Day 49's training data only covers early morning hours (0:00 - 2:00 AM), standard GBDT models (LightGBM, XGBoost, CatBoost) struggle to extrapolate to daytime peak demand values on the same day. 

---

## 🛠️ Solutions Implemented

We designed and iterated on two primary architectural solutions:

### 1. Two-Stage Linear Extrapolation Model (Hybrid)
To bypass GBDT extrapolation limits, we build a hybrid model:
*   **Stage 1**: Train GBDT models on Day 48 (24 hours) using out-of-fold target encodings to predict a baseline profile for Day 49.
*   **Lag Mapping**: Map Day 48 demand at the same `(geohash, time_slot)` to Day 49 as a physical lag feature (`demand_prev_day`).
*   **Stage 2**: Fit a highly regularized Ridge Regression model ($\alpha=50.0$) with degree-2 polynomial interactions on Day 49 train data (hours 0-2) to scale Stage 1 predictions and lag features smoothly for the rest of Day 49.

### 2. Ultimate GPU-Accelerated Ensemble Pipeline (Current Best)
Our highest-performing model uses a robust, **unleaked** target-encoding scheme and a powerful blend of GBDT regressors trained with GPU acceleration:
*   **Out-Of-Fold Target Encodings**: Target encodings for `geohash`, `(geohash, hour)`, and `(geohash, time_slot)` are computed strictly using 5-fold cross-validation on Day 48 to prevent target leakage.
*   **CatBoost Regressor (GPU)**: 10-fold cross-validation with 2,500 trees (`depth=8`, `learning_rate=0.04`).
*   **XGBoost Regressor (CUDA)**: 10-fold cross-validation with 3,000 trees (`max_depth=8`, `learning_rate=0.02`, `reg_lambda=2.0`).
*   **Ensemble Blend**: 15% CatBoost + 85% XGBoost, clipped to `[0, 1]`.

```mermaid
graph TD
    A[train.csv / test.csv] --> B[Split by Day]
    B1[Day 48 - Full 24 Hours] --> C[Lookup Table: geohash + time_slot]
    B2[Day 49 / Test Set] --> D[Map demand_prev_day from Lookup]
    C --> D
    B1 --> E[Compute Out-Of-Fold Target Encodings:<br>te_geo, te_geo_hour, te_geo_ts]
    E --> F[Map Encodings to train/test]
    D --> G[Feature Engineering:<br>RoadType x Lanes, geo_zone_4, geo_zone_5, Geo_Hour]
    F --> G
    G --> H1[10-Fold CatBoost Regressor GPU]
    G --> H2[10-Fold XGBoost Regressor CUDA]
    H1 --> I1[CatBoost Predictions]
    H2 --> I2[XGBoost Predictions]
    I1 --> J[15 / 85 Ensemble Blend]
    I2 --> J
    J --> K[Clip to [0,1]]
    K --> L[Submission CSV]
```

---

## 📈 Leaderboard Progression

| Version | Model / Strategy | Local Day 49 OOF $R^2$ | Public Leaderboard Score | Status |
| :--- | :--- | :---: | :---: | :---: |
| **T1** | Baseline LGBM | — | 83.270 | Checked in |
| **T2** | Global GBDT (LGBM + XGBoost) | — | 85.388 | Checked in |
| **T7** | Two-Stage Ridge Blend (35% Ridge / 65% GBDT) | — | 87.830 | Checked in |
| **T8** | 10-Fold CatBoost + Lag | 93.04% | 91.512 | Checked in |
| **T9** | CatBoost + LightGBM Blend | 95.24% | **91.590** | Checked in |
| **T10** | CatBoost + Leaked Target Encodings | 98.71% (Leaked) | 90.560 | Degraded (leak identified) |
| **T11** | Ultimate GPU CatBoost/XGB Ensemble + Unleaked TE | **95.70%** | **TBD** | **Best Submission** |

---

## 📁 Repository Structure

*   `train_cuda_ensemble_final.py`: Core pipeline training script. Configured to run on CUDA-enabled GPU.
*   `train_catboost_te_final.py`: Single CatBoost model pipeline with OOF Target Encodings.
*   `generate_all_blends.py` / `generate_blends_v3.py`: Scripts used to generate weighted predictions between Two-Stage and Global models.
*   `approach.txt`: Concise strategy summary document.
*   `requirements.txt`: Python package requirements.
