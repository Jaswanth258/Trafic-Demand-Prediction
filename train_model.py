"""
Traffic Demand Prediction - Gridlock Hackathon 2.0 (v3)
========================================================
Key improvements over v1 (scored 83.27):
- Day-specific target encoding (day 48 vs day 49 patterns)
- Bayesian smoothed target encoding (less overfitting)
- Log1p target transform (better for right-skewed demand)
- Frequency encoding (geohash popularity)
- Stronger regularization + more iterations
- Optimal ensemble weight search
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
np.random.seed(SEED)

# Use log1p transform on target
USE_LOG_TARGET = True


# ============================================================
# Geohash Decoder
# ============================================================
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_BASE32_MAP = {c: i for i, c in enumerate(_BASE32)}

def decode_geohash(gh_str):
    bits = []
    for char in gh_str.lower():
        idx = _BASE32_MAP.get(char, 0)
        bits.extend([(idx >> (4 - i)) & 1 for i in range(5)])
    lon_bits, lat_bits = bits[0::2], bits[1::2]
    def _b2c(b, lo, hi):
        for bit in b:
            mid = (lo + hi) / 2
            if bit: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    return _b2c(lat_bits, -90, 90), _b2c(lon_bits, -180, 180)


# ============================================================
# Bayesian Smoothed Target Encoding
# ============================================================
def bayesian_te(train_data, target, group_cols, smoothing=20):
    """Returns dict mapping group -> smoothed mean."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    global_mean = target.mean()
    temp = train_data[group_cols].copy()
    temp["_target"] = target
    
    agg = temp.groupby(group_cols)["_target"].agg(["mean", "count"])
    agg["smoothed"] = (
        (agg["count"] * agg["mean"] + smoothing * global_mean) /
        (agg["count"] + smoothing)
    )
    return agg["smoothed"], global_mean


def apply_te(df, smoothed_series, group_cols, global_mean, col_name):
    """Apply pre-computed target encoding to a dataframe."""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    result = df[group_cols].copy()
    merged = result.merge(
        smoothed_series.rename(col_name).reset_index(),
        on=group_cols, how="left"
    )
    return merged[col_name].fillna(global_mean).values


# ============================================================
# Load Data
# ============================================================
def load_data():
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    print(f"  Train: {train.shape}  |  Test: {test.shape}")
    print(f"  Target: mean={train['demand'].mean():.6f} std={train['demand'].std():.6f}")
    print(f"  Train days: {sorted(train['day'].unique())}")
    print(f"  Test days:  {sorted(test['day'].unique())}")
    
    common = len(set(train["geohash"]) & set(test["geohash"]))
    print(f"  Common geohashes: {common}/{test['geohash'].nunique()}")
    
    # Day distribution
    for d in sorted(train["day"].unique()):
        n = len(train[train["day"] == d])
        print(f"  Train day {d}: {n} rows")
    
    return train, test


# ============================================================
# Feature Engineering
# ============================================================
def build_features(train, test):
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)
    
    target = train["demand"].values.astype(np.float64)
    if USE_LOG_TARGET:
        target_transformed = np.log1p(target)
        print(f"  Using log1p transform: mean={target_transformed.mean():.6f}")
    else:
        target_transformed = target.copy()
    
    test_idx = test["Index"].copy()
    n_train = len(train)
    
    # Combine for base features
    combined = pd.concat([
        train.drop("demand", axis=1),
        test
    ], axis=0, ignore_index=True)
    
    # --- Geohash ---
    print("  [1/6] Geohash features...")
    coords = combined["geohash"].apply(decode_geohash)
    combined["latitude"] = coords.apply(lambda c: c[0])
    combined["longitude"] = coords.apply(lambda c: c[1])
    combined["geohash_5"] = combined["geohash"].str[:5]
    combined["geohash_4"] = combined["geohash"].str[:4]
    
    for col in ["geohash", "geohash_5", "geohash_4"]:
        le = LabelEncoder()
        combined[col + "_enc"] = le.fit_transform(combined[col].astype(str))
    
    # Frequency encoding (how popular is each geohash)
    geo_freq = combined["geohash"].value_counts().to_dict()
    combined["geohash_freq"] = combined["geohash"].map(geo_freq)
    geo5_freq = combined["geohash_5"].value_counts().to_dict()
    combined["geohash5_freq"] = combined["geohash_5"].map(geo5_freq)
    
    # --- Temporal ---
    print("  [2/6] Temporal features...")
    ts = combined["timestamp"].astype(str).str.split(":", expand=True)
    combined["hour"] = ts[0].astype(int)
    combined["minute"] = ts[1].astype(int)
    combined["time_slot"] = combined["hour"] * 4 + combined["minute"] // 15
    combined["minutes_since_midnight"] = combined["hour"] * 60 + combined["minute"]
    
    combined["hour_sin"] = np.sin(2 * np.pi * combined["hour"] / 24)
    combined["hour_cos"] = np.cos(2 * np.pi * combined["hour"] / 24)
    combined["slot_sin"] = np.sin(2 * np.pi * combined["time_slot"] / 96)
    combined["slot_cos"] = np.cos(2 * np.pi * combined["time_slot"] / 96)
    
    combined["is_rush_morning"] = ((combined["hour"] >= 8) & (combined["hour"] <= 10)).astype(int)
    combined["is_rush_evening"] = ((combined["hour"] >= 17) & (combined["hour"] <= 19)).astype(int)
    combined["is_rush_hour"] = (combined["is_rush_morning"] | combined["is_rush_evening"]).astype(int)
    combined["is_night"] = ((combined["hour"] >= 22) | (combined["hour"] <= 5)).astype(int)
    combined["is_midday"] = ((combined["hour"] >= 11) & (combined["hour"] <= 15)).astype(int)
    
    # --- Categorical ---
    print("  [3/6] Categorical features...")
    combined["RoadType"] = combined["RoadType"].fillna("Unknown")
    road_map = {"Unknown": 0, "Residential": 1, "Street": 2, "Highway": 3}
    combined["RoadType_enc"] = combined["RoadType"].map(road_map).fillna(0).astype(int)
    combined["LargeVehicles_enc"] = (combined["LargeVehicles"] == "Allowed").astype(int)
    combined["Landmarks_enc"] = (combined["Landmarks"] == "Yes").astype(int)
    combined["Weather"] = combined["Weather"].fillna("Unknown")
    weather_map = {"Unknown": 0, "Sunny": 1, "Rainy": 2, "Foggy": 3, "Snowy": 4}
    combined["Weather_enc"] = combined["Weather"].map(weather_map).fillna(0).astype(int)
    combined["Temperature"] = combined["Temperature"].fillna(combined["Temperature"].median())
    
    # --- Interactions ---
    print("  [4/6] Interaction features...")
    combined["road_x_lanes"] = combined["RoadType_enc"] * 10 + combined["NumberofLanes"]
    combined["road_x_hour"] = combined["RoadType_enc"] * 100 + combined["hour"]
    combined["weather_x_hour"] = combined["Weather_enc"] * 100 + combined["hour"]
    combined["road_x_weather"] = combined["RoadType_enc"] * 10 + combined["Weather_enc"]
    combined["lat_x_lon"] = combined["latitude"] * combined["longitude"]
    combined["lanes_x_large"] = combined["NumberofLanes"] * combined["LargeVehicles_enc"]
    combined["landmark_x_rush"] = combined["Landmarks_enc"] * combined["is_rush_hour"]
    combined["road_x_rush"] = combined["RoadType_enc"] * combined["is_rush_hour"]
    
    # Split
    train_df = combined.iloc[:n_train].copy().reset_index(drop=True)
    test_df = combined.iloc[n_train:].copy().reset_index(drop=True)
    
    # --- Target Encoding (Bayesian smoothed) ---
    print("  [5/6] Bayesian target encoding...")
    
    # We use the TRANSFORMED target for target encoding
    te_configs = [
        # (group_cols, prefix, smoothing)
        ("geohash_enc",                     "te_geo",      20),
        ("geohash_5_enc",                   "te_geo5",     20),
        ("time_slot",                       "te_slot",     15),
        ("hour",                            "te_hour",     15),
        ("RoadType_enc",                    "te_road",     15),
        ("Weather_enc",                     "te_weath",    15),
        ("NumberofLanes",                   "te_lanes",    15),
        (["geohash_enc", "hour"],           "te_geo_h",    10),
        (["geohash_enc", "time_slot"],      "te_geo_ts",    5),
        (["geohash_5_enc", "hour"],         "te_g5_h",     10),
        (["geohash_5_enc", "time_slot"],    "te_g5_ts",     5),
        (["RoadType_enc", "hour"],          "te_road_h",   10),
        (["RoadType_enc", "time_slot"],     "te_road_ts",   5),
        (["Weather_enc", "hour"],           "te_weath_h",  10),
        (["RoadType_enc", "Weather_enc"],   "te_road_w",   10),
        (["geohash_enc", "RoadType_enc"],   "te_geo_r",    10),
    ]
    
    te_col_names = []
    for group_cols, prefix, smooth in te_configs:
        col_name = f"{prefix}_mean"
        smoothed, gmean = bayesian_te(train_df, target_transformed, group_cols, smooth)
        train_df[col_name] = apply_te(train_df, smoothed, group_cols, gmean, col_name)
        test_df[col_name] = apply_te(test_df, smoothed, group_cols, gmean, col_name)
        te_col_names.append(col_name)
    
    # --- Day-specific target encoding (NEW) ---
    print("  [6/6] Day-specific target encoding...")
    
    # Create day-specific encodings using original train data
    train_with_day = train_df.copy()
    train_with_day["_day"] = train["day"].values
    train_with_day["_target"] = target_transformed
    
    for day_val in sorted(train["day"].unique()):
        day_mask = train_with_day["_day"] == day_val
        day_data = train_with_day[day_mask]
        day_target = target_transformed[day_mask.values]
        
        if len(day_data) < 100:
            continue
        
        # Geohash mean demand on this specific day
        col = f"te_geo_day{day_val}"
        sm, gm = bayesian_te(day_data, day_target, "geohash_enc", smoothing=10)
        train_df[col] = apply_te(train_df, sm, "geohash_enc", gm, col)
        test_df[col] = apply_te(test_df, sm, "geohash_enc", gm, col)
        te_col_names.append(col)
        
        # Geohash x hour on this day
        col2 = f"te_geo_h_day{day_val}"
        sm2, gm2 = bayesian_te(day_data, day_target, ["geohash_enc", "hour"], smoothing=5)
        train_df[col2] = apply_te(train_df, sm2, ["geohash_enc", "hour"], gm2, col2)
        test_df[col2] = apply_te(test_df, sm2, ["geohash_enc", "hour"], gm2, col2)
        te_col_names.append(col2)
        
        # Geohash x time_slot on this day
        col3 = f"te_geo_ts_day{day_val}"
        sm3, gm3 = bayesian_te(day_data, day_target, ["geohash_enc", "time_slot"], smoothing=3)
        train_df[col3] = apply_te(train_df, sm3, ["geohash_enc", "time_slot"], gm3, col3)
        test_df[col3] = apply_te(test_df, sm3, ["geohash_enc", "time_slot"], gm3, col3)
        te_col_names.append(col3)
    
    # --- Spatial neighborhood ---
    geo5_mean = train_df.groupby("geohash_5_enc").apply(
        lambda x: target_transformed[x.index].mean()
    )
    train_df["spatial_neigh"] = train_df["geohash_5_enc"].map(geo5_mean).fillna(target_transformed.mean())
    test_df["spatial_neigh"] = test_df["geohash_5_enc"].map(geo5_mean).fillna(target_transformed.mean())
    
    train_df["geo_vs_neigh"] = train_df["te_geo_mean"] - train_df["spatial_neigh"]
    test_df["geo_vs_neigh"] = test_df["te_geo_mean"] - test_df["spatial_neigh"]
    
    # --- Final feature list ---
    drop_cols = [
        "Index", "geohash", "geohash_4", "geohash_5",
        "RoadType", "LargeVehicles", "Landmarks", "Weather",
        "timestamp", "day", "_day", "_target",
    ]
    feature_cols = sorted([c for c in train_df.columns if c not in drop_cols])
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    
    print(f"\n  Features: {len(feature_cols)}")
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    
    return X_train, X_test, target_transformed.astype(np.float32), target, test_idx, feature_cols


# ============================================================
# GPU Detection
# ============================================================
def _try_gpu_lgb():
    try:
        m = lgb.LGBMRegressor(device="gpu", n_estimators=2, verbose=-1)
        m.fit(np.random.rand(20, 3).astype(np.float32),
              np.random.rand(20).astype(np.float32))
        return True
    except Exception:
        return False

def _try_gpu_xgb():
    try:
        m = xgb.XGBRegressor(tree_method="hist", device="cuda",
                             n_estimators=2, verbosity=0)
        m.fit(np.random.rand(20, 3).astype(np.float32),
              np.random.rand(20).astype(np.float32))
        return True
    except Exception:
        return False


# ============================================================
# Training
# ============================================================
def train_models(X_train, X_test, y_train, y_train_original, feature_cols):
    print("\n" + "=" * 60)
    print("STEP 3: Model Training")
    print("=" * 60)
    
    gpu_lgb = _try_gpu_lgb()
    gpu_xgb = _try_gpu_xgb()
    print(f"  LGB GPU: {gpu_lgb}  |  XGB CUDA: {gpu_xgb}")
    
    # --- LightGBM ---
    lgb_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "learning_rate": 0.02,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "min_child_samples": 25,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "max_bin": 255,
        "n_estimators": 10000,
        "verbose": -1,
        "random_state": SEED,
    }
    if gpu_lgb:
        lgb_params["device"] = "gpu"
        lgb_params["gpu_use_dp"] = False
    
    # --- XGBoost ---
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 8,
        "learning_rate": 0.02,
        "subsample": 0.75,
        "colsample_bytree": 0.7,
        "min_child_weight": 8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "n_estimators": 10000,
        "tree_method": "hist",
        "random_state": SEED,
        "verbosity": 0,
        "early_stopping_rounds": 300,
    }
    if gpu_xgb:
        xgb_params["device"] = "cuda"
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    lgb_oof = np.zeros(len(X_train))
    xgb_oof = np.zeros(len(X_train))
    lgb_test = np.zeros(len(X_test))
    xgb_test = np.zeros(len(X_test))
    lgb_models, xgb_models = [], []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\n  --- FOLD {fold+1}/{N_FOLDS} ---")
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(1000)],
        )
        lgb_pred = lgb_model.predict(X_val)
        lgb_oof[val_idx] = lgb_pred
        lgb_test += lgb_model.predict(X_test) / N_FOLDS
        lgb_models.append(lgb_model)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=1000)
        xgb_pred = xgb_model.predict(X_val)
        xgb_oof[val_idx] = xgb_pred
        xgb_test += xgb_model.predict(X_test) / N_FOLDS
        xgb_models.append(xgb_model)
        
        # Score in ORIGINAL space (for comparison)
        if USE_LOG_TARGET:
            lgb_orig = np.expm1(lgb_pred)
            xgb_orig = np.expm1(xgb_pred)
            y_val_orig = y_train_original[val_idx]
        else:
            lgb_orig = lgb_pred
            xgb_orig = xgb_pred
            y_val_orig = y_val
        
        lgb_r2 = r2_score(y_val_orig, lgb_orig)
        xgb_r2 = r2_score(y_val_orig, xgb_orig)
        print(f"  LGB: R2={lgb_r2:.6f} Score={max(0,100*lgb_r2):.2f} (iter={lgb_model.best_iteration_})")
        print(f"  XGB: R2={xgb_r2:.6f} Score={max(0,100*xgb_r2):.2f} (iter={xgb_model.best_iteration})")
    
    # --- Overall OOF scores ---
    print("\n" + "=" * 60)
    print("STEP 4: Ensemble Results")
    print("=" * 60)
    
    if USE_LOG_TARGET:
        lgb_oof_orig = np.expm1(lgb_oof)
        xgb_oof_orig = np.expm1(xgb_oof)
    else:
        lgb_oof_orig = lgb_oof
        xgb_oof_orig = xgb_oof
    
    lgb_r2 = r2_score(y_train_original, lgb_oof_orig)
    xgb_r2 = r2_score(y_train_original, xgb_oof_orig)
    print(f"  LGB OOF: R2={lgb_r2:.6f}  Score={max(0,100*lgb_r2):.2f}")
    print(f"  XGB OOF: R2={xgb_r2:.6f}  Score={max(0,100*xgb_r2):.2f}")
    
    # Optimal blend in original space
    best_r2 = -999
    best_w = 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        blend = w * lgb_oof_orig + (1-w) * xgb_oof_orig
        r2 = r2_score(y_train_original, blend)
        if r2 > best_r2:
            best_r2 = r2
            best_w = w
    
    # Also try blending in log space then converting
    best_r2_log = -999
    best_w_log = 0.5
    for w in np.arange(0.0, 1.01, 0.05):
        blend_log = w * lgb_oof + (1-w) * xgb_oof
        if USE_LOG_TARGET:
            blend_orig = np.expm1(blend_log)
        else:
            blend_orig = blend_log
        r2 = r2_score(y_train_original, blend_orig)
        if r2 > best_r2_log:
            best_r2_log = r2
            best_w_log = w
    
    print(f"\n  Blend in original space: w_lgb={best_w:.2f} R2={best_r2:.6f} Score={max(0,100*best_r2):.2f}")
    print(f"  Blend in log space:      w_lgb={best_w_log:.2f} R2={best_r2_log:.6f} Score={max(0,100*best_r2_log):.2f}")
    
    # Use the better blending strategy
    if best_r2_log >= best_r2:
        print("  -> Using log-space blending")
        final_log = best_w_log * lgb_test + (1-best_w_log) * xgb_test
        if USE_LOG_TARGET:
            final_preds = np.expm1(final_log)
        else:
            final_preds = final_log
        best_score = best_r2_log
    else:
        print("  -> Using original-space blending")
        if USE_LOG_TARGET:
            lgb_test_orig = np.expm1(lgb_test)
            xgb_test_orig = np.expm1(xgb_test)
        else:
            lgb_test_orig = lgb_test
            xgb_test_orig = xgb_test
        final_preds = best_w * lgb_test_orig + (1-best_w) * xgb_test_orig
        best_score = best_r2
    
    final_preds = np.clip(final_preds, 0, 1)
    
    print(f"\n  Ensemble Score (OOF): {max(0, 100*best_score):.2f}")
    print(f"  Preds: mean={final_preds.mean():.6f} std={final_preds.std():.6f} "
          f"min={final_preds.min():.8f} max={final_preds.max():.6f}")
    
    # Feature importance
    print(f"\n  Top 15 features (LightGBM):")
    imp = np.mean([m.feature_importances_ for m in lgb_models], axis=0)
    ranking = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)
    for i, (feat, score) in enumerate(ranking[:15]):
        bar = "#" * int(score / max(imp) * 25)
        print(f"    {i+1:2d}. {feat:35s} {score:8.1f}  {bar}")
    
    return final_preds


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    
    train, test = load_data()
    X_train, X_test, y_transformed, y_original, test_idx, feat_cols = \
        build_features(train, test)
    
    predictions = train_models(X_train, X_test, y_transformed, y_original, feat_cols)
    
    # --- Submission ---
    print("\n" + "=" * 60)
    print("STEP 5: Submission")
    print("=" * 60)
    
    sub = pd.DataFrame({"Index": test_idx.values, "demand": predictions})
    out_path = os.path.join(OUTPUT_DIR, "T2_submission.csv")
    sub.to_csv(out_path, index=False)
    
    print(f"  Saved: {out_path}")
    print(f"  Shape: {sub.shape}")
    print(sub.head(10).to_string(index=False))
    
    assert len(sub) == 41778, f"Row count: {len(sub)}"
    assert sub["demand"].between(0, 1).all(), "Out of range!"
    assert sub.columns.tolist() == ["Index", "demand"]
    print("\n  [OK] All checks passed")
    
    print(f"\nDONE -- {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
