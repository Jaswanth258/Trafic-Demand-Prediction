import pandas as pd
import numpy as np
import os
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_DIR = r"c:\Users\jaswa\OneDrive\Desktop\TDP\dataset"
OUTPUT_DIR = r"c:\Users\jaswa\OneDrive\Desktop\TDP"

print("="*60)
print("Loading datasets...")
print("="*60)
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
t2_sub = pd.read_csv(os.path.join(OUTPUT_DIR, "T2_submission.csv"))

# Parse timestamp
for df in [train, test]:
    ts = df['timestamp'].str.split(':', expand=True)
    df['hour'] = ts[0].astype(int)
    df['minute'] = ts[1].astype(int)
    df['time_slot'] = df['hour'] * 4 + df['minute'] // 15

# Split day 48 and 49
train_48 = train[train['day'] == 48].copy().reset_index(drop=True)
train_49 = train[train['day'] == 49].copy().reset_index(drop=True)

# Decode geohash
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

print("Feature engineering...")
for df in [train_48, train_49, test]:
    coords = df['geohash'].apply(decode_geohash)
    df['latitude'] = coords.apply(lambda c: c[0])
    df['longitude'] = coords.apply(lambda c: c[1])
    
    road_map = {"Residential": 1, "Street": 2, "Highway": 3}
    df['RoadType_enc'] = df['RoadType'].map(road_map).fillna(0).astype(int)
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == "Allowed").astype(int)
    df['Landmarks_enc'] = (df['Landmarks'] == "Yes").astype(int)
    weather_map = {"Sunny": 1, "Rainy": 2, "Foggy": 3, "Snowy": 4}
    df['Weather_enc'] = df['Weather'].map(weather_map).fillna(0).astype(int)
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    
    # Cyclical time
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['slot_sin'] = np.sin(2 * np.pi * df['time_slot'] / 96)
    df['slot_cos'] = np.cos(2 * np.pi * df['time_slot'] / 96)

# Construct Target Encodings on Day 48
print("Computing target encodings on Day 48...")
global_mean_48 = train_48['demand'].mean()

agg_geo = train_48.groupby('geohash')['demand'].agg(['mean', 'count'])
te_geo_full = (agg_geo['count'] * agg_geo['mean'] + 15 * global_mean_48) / (agg_geo['count'] + 15)

agg_geo_h = train_48.groupby(['geohash', 'hour'])['demand'].agg(['mean', 'count'])
te_geo_h_full = (agg_geo_h['count'] * agg_geo_h['mean'] + 15 * global_mean_48) / (agg_geo_h['count'] + 15)

agg_geo_ts = train_48.groupby(['geohash', 'time_slot'])['demand'].agg(['mean', 'count'])
te_geo_ts_full = (agg_geo_ts['count'] * agg_geo_ts['mean'] + 15 * global_mean_48) / (agg_geo_ts['count'] + 15)

# OOF target encodings for Stage 1 training
kf_48 = KFold(n_splits=5, shuffle=True, random_state=42)
train_48['te_geo'] = np.nan
train_48['te_geo_h'] = np.nan
train_48['te_geo_ts'] = np.nan

def compute_smoothed_te(tr_df, val_df, group_cols, smoothing=15):
    agg = tr_df.groupby(group_cols)['demand'].agg(['mean', 'count'])
    smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean_48) / (agg['count'] + smoothing)
    smoothed = smoothed.rename('smoothed_value')
    if isinstance(group_cols, list):
        merged = val_df[group_cols].merge(smoothed.reset_index(), on=group_cols, how='left')
        return merged['smoothed_value'].fillna(global_mean_48).values
    else:
        return val_df[group_cols].map(smoothed).fillna(global_mean_48).values

for fold, (tr_idx, val_idx) in enumerate(kf_48.split(train_48)):
    tr_data = train_48.iloc[tr_idx]
    val_data = train_48.iloc[val_idx]
    
    train_48.loc[val_idx, 'te_geo'] = compute_smoothed_te(tr_data, val_data, 'geohash')
    train_48.loc[val_idx, 'te_geo_h'] = compute_smoothed_te(tr_data, val_data, ['geohash', 'hour'])
    train_48.loc[val_idx, 'te_geo_ts'] = compute_smoothed_te(tr_data, val_data, ['geohash', 'time_slot'])

# Map target encodings to Day 49 and Test using FULL Day 48 stats
train_49['te_geo'] = train_49['geohash'].map(te_geo_full).fillna(global_mean_48)
test['te_geo'] = test['geohash'].map(te_geo_full).fillna(global_mean_48)

train_49 = train_49.merge(te_geo_h_full.rename('te_geo_h').reset_index(), on=['geohash', 'hour'], how='left')
train_49['te_geo_h'] = train_49['te_geo_h'].fillna(global_mean_48)
test = test.merge(te_geo_h_full.rename('te_geo_h').reset_index(), on=['geohash', 'hour'], how='left')
test['te_geo_h'] = test['te_geo_h'].fillna(global_mean_48)

train_49 = train_49.merge(te_geo_ts_full.rename('te_geo_ts').reset_index(), on=['geohash', 'time_slot'], how='left')
train_49['te_geo_ts'] = train_49['te_geo_ts'].fillna(global_mean_48)
test = test.merge(te_geo_ts_full.rename('te_geo_ts').reset_index(), on=['geohash', 'time_slot'], how='left')
test['te_geo_ts'] = test['te_geo_ts'].fillna(global_mean_48)

# Stage 1 Features
features_stage1 = [
    'latitude', 'longitude', 'RoadType_enc', 'NumberofLanes',
    'LargeVehicles_enc', 'Landmarks_enc', 'Weather_enc', 'Temperature',
    'hour_sin', 'hour_cos', 'slot_sin', 'slot_cos',
    'te_geo', 'te_geo_h', 'te_geo_ts'
]

# Train Stage 1 Models on Day 48
print("Training Stage 1 LightGBM and XGBoost...")
X_48 = train_48[features_stage1].values
y_48 = train_48['demand'].values

lgb_model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
lgb_model.fit(X_48, y_48)

xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)
xgb_model.fit(X_48, y_48)

# Predict Stage 1 for Day 49 and Test
print("Generating Stage 1 predictions...")
train_49['pred_stage1_lgb'] = lgb_model.predict(train_49[features_stage1].values)
train_49['pred_stage1_xgb'] = xgb_model.predict(train_49[features_stage1].values)

test['pred_stage1_lgb'] = lgb_model.predict(test[features_stage1].values)
test['pred_stage1_xgb'] = xgb_model.predict(test[features_stage1].values)

# Map demand_prev_day (Day 48 demand)
print("Mapping demand_prev_day...")
lookup_48 = train_48.set_index(['geohash', 'time_slot'])['demand'].to_dict()
geohash_mean_48 = train_48.groupby('geohash')['demand'].mean().to_dict()

for df in [train_49, test]:
    df['demand_prev_day'] = df.apply(
        lambda row: lookup_48.get((row['geohash'], row['time_slot']), np.nan),
        axis=1
    )
    df['geohash_mean_48'] = df['geohash'].map(geohash_mean_48).fillna(global_mean_48)
    df['demand_prev_day'] = df['demand_prev_day'].fillna(df['geohash_mean_48'])

# Stage 2 features
features_stage2 = [
    'pred_stage1_lgb', 'pred_stage1_xgb', 'demand_prev_day',
    'latitude', 'longitude', 'RoadType_enc', 'NumberofLanes',
    'LargeVehicles_enc', 'Landmarks_enc', 'Weather_enc', 'Temperature'
]

# Fit Stage 2 Ridge model on Day 49 train with optimized alpha=5.0
print("Fitting Stage 2 Ridge regression model (alpha=5.0)...")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_49_stage2 = poly.fit_transform(train_49[features_stage2].values)
y_49 = train_49['demand'].values

# Ridge standardization parameters
mean_stage2 = X_49_stage2.mean(axis=0)
std_stage2 = X_49_stage2.std(axis=0) + 1e-8
X_49_scaled = (X_49_stage2 - mean_stage2) / std_stage2

# Fit Ridge
ridge = Ridge(alpha=5.0)
ridge.fit(X_49_scaled, y_49)

# Predict Stage 2 for test
print("Predicting test set using Stage 2...")
X_test_stage2 = poly.transform(test[features_stage2].values)
X_test_scaled = (X_test_stage2 - mean_stage2) / std_stage2
pred_twostage = ridge.predict(X_test_scaled)
pred_twostage = np.clip(pred_twostage, 0, 1)

# Blending weights configurations
blends = {
    "T4": 0.60, # 60% Two-Stage, 40% GBDT
    "T5": 0.70, # 70% Two-Stage, 30% GBDT
    "T6": 0.65  # 65% Two-Stage, 35% GBDT
}

for name, w in blends.items():
    print(f"Generating {name} submission with weight {w:.2f} Two-Stage...")
    final_preds = w * pred_twostage + (1 - w) * t2_sub['demand'].values
    final_preds = np.clip(final_preds, 0, 1)
    
    sub = pd.DataFrame({
        "Index": test['Index'].values,
        "demand": final_preds
    })
    
    sub_path = os.path.join(OUTPUT_DIR, f"{name}_submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")
    assert len(sub) == 41778, "Incorrect row count!"

print("="*60)
print("All blend submissions generated successfully!")
print("="*60)
