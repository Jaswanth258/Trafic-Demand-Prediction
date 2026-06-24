import pandas as pd
import numpy as np
import os
import zipfile
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import xgboost as xgb

warnings.filterwarnings('ignore')

print("RUNNING ULTIMATE GPU-ACCELERATED CATBOOST + XGBOOST ENSEMBLE PIPELINE FOR T11...")

# ==========================================
# 1. LOAD DATA
# ==========================================
DATA_DIR = r"dataset"
OUTPUT_DIR = r"."

train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
test_index = test['Index']

# Parse timestamp into Hour and Minute to get time_slot
for df in [train, test]:
    ts = df['timestamp'].str.split(':', expand=True)
    df['Hour'] = ts[0].astype(int)
    df['Minute'] = ts[1].astype(int)
    df['time_slot'] = df['Hour'] * 4 + df['Minute'] // 15

# Split day 48 and 49
train_48 = train[train['day'] == 48].copy()

# ==========================================
# 2. LAG FEATURE MAPPING (demand_prev_day)
# ==========================================
print("Mapping lag features (demand_prev_day)...")

# Fast lookup for day 48
lookup_48 = train_48[['geohash', 'time_slot', 'demand']].rename(columns={'demand': 'demand_prev_day'})
lookup_48 = lookup_48.drop_duplicates(subset=['geohash', 'time_slot'])

# Merge to train and test
train = train.merge(lookup_48, on=['geohash', 'time_slot'], how='left')
test = test.merge(lookup_48, on=['geohash', 'time_slot'], how='left')

# For day 48 rows, set demand_prev_day to NaN (we only lag for Day 49 / Test)
train.loc[train['day'] == 48, 'demand_prev_day'] = np.nan

# Impute missing demand_prev_day with geohash mean on Day 48
geohash_mean_48 = train_48.groupby('geohash')['demand'].mean().to_dict()
global_mean_48 = train_48['demand'].mean()

train['geohash_mean_48'] = train['geohash'].map(geohash_mean_48).fillna(global_mean_48)
test['geohash_mean_48'] = test['geohash'].map(geohash_mean_48).fillna(global_mean_48)

train.loc[train['day'] == 49, 'demand_prev_day'] = train.loc[train['day'] == 49, 'demand_prev_day'].fillna(train['geohash_mean_48'])
test['demand_prev_day'] = test['demand_prev_day'].fillna(test['geohash_mean_48'])

# ==========================================
# 3. UNLEAKED TARGET ENCODINGS FROM DAY 48
# ==========================================
print("Computing and mapping target encodings...")

# Helper to compute smoothed TE for a fold
def get_smoothed_te_series(tr_df, group_cols, target_col, smoothing=15):
    agg = tr_df.groupby(group_cols)[target_col].agg(['mean', 'count'])
    smoothed = (agg['count'] * agg['mean'] + smoothing * global_mean_48) / (agg['count'] + smoothing)
    return smoothed

# Compute OOF target encodings on train_48 to prevent leakage
train_48_split = train[train['day'] == 48].copy().reset_index(drop=True)
kf_te = KFold(n_splits=5, shuffle=True, random_state=42)

oof_geo = np.zeros(len(train_48_split))
oof_geo_hour = np.zeros(len(train_48_split))
oof_geo_ts = np.zeros(len(train_48_split))

for fold, (tr_idx, val_idx) in enumerate(kf_te.split(train_48_split)):
    tr_fold = train_48_split.iloc[tr_idx]
    val_fold = train_48_split.iloc[val_idx]
    
    # 1. geohash
    te_series_geo = get_smoothed_te_series(tr_fold, 'geohash', 'demand', smoothing=20)
    oof_geo[val_idx] = val_fold['geohash'].map(te_series_geo).fillna(global_mean_48).values
    
    # 2. geohash + Hour
    te_series_hour = get_smoothed_te_series(tr_fold, ['geohash', 'Hour'], 'demand', smoothing=15)
    val_fold_mapped = val_fold.merge(te_series_hour.rename('te_geo_hour').reset_index(), on=['geohash', 'Hour'], how='left')
    oof_geo_hour[val_idx] = val_fold_mapped['te_geo_hour'].fillna(global_mean_48).values
    
    # 3. geohash + time_slot
    te_series_ts = get_smoothed_te_series(tr_fold, ['geohash', 'time_slot'], 'demand', smoothing=10)
    val_fold_mapped_ts = val_fold.merge(te_series_ts.rename('te_geo_ts').reset_index(), on=['geohash', 'time_slot'], how='left')
    oof_geo_ts[val_idx] = val_fold_mapped_ts['te_geo_ts'].fillna(global_mean_48).values

# Assign back to train_48_split
train_48_split['te_geo'] = oof_geo
train_48_split['te_geo_hour'] = oof_geo_hour
train_48_split['te_geo_ts'] = oof_geo_ts

# Now map the FULL Day 48 encodings to Day 49 and Test (no OOF needed for future sets since Day 48 is entirely in the past)
full_series_geo = get_smoothed_te_series(train_48, 'geohash', 'demand', smoothing=20)
full_series_hour = get_smoothed_te_series(train_48, ['geohash', 'Hour'], 'demand', smoothing=15)
full_series_ts = get_smoothed_te_series(train_48, ['geohash', 'time_slot'], 'demand', smoothing=10)

# Merge back train_48 to train
for col in ['te_geo', 'te_geo_hour', 'te_geo_ts']:
    train.loc[train['day'] == 48, col] = train_48_split[col].values

# Map to train_49
train_49 = train[train['day'] == 49].copy().reset_index(drop=True)
train_49 = train_49.drop(columns=['te_geo', 'te_geo_hour', 'te_geo_ts'], errors='ignore')
train_49['te_geo'] = train_49['geohash'].map(full_series_geo).fillna(global_mean_48)

train_49_mapped_h = train_49.merge(full_series_hour.rename('te_geo_hour').reset_index(), on=['geohash', 'Hour'], how='left')
train_49['te_geo_hour'] = train_49_mapped_h['te_geo_hour'].fillna(global_mean_48).values

train_49_mapped_ts = train_49.merge(full_series_ts.rename('te_geo_ts').reset_index(), on=['geohash', 'time_slot'], how='left')
train_49['te_geo_ts'] = train_49_mapped_ts['te_geo_ts'].fillna(global_mean_48).values

for col in ['te_geo', 'te_geo_hour', 'te_geo_ts']:
    train.loc[train['day'] == 49, col] = train_49[col].values

# Map to test
test = test.drop(columns=['te_geo', 'te_geo_hour', 'te_geo_ts'], errors='ignore')
test['te_geo'] = test['geohash'].map(full_series_geo).fillna(global_mean_48)

test_mapped_h = test.merge(full_series_hour.rename('te_geo_hour').reset_index(), on=['geohash', 'Hour'], how='left')
test['te_geo_hour'] = test_mapped_h['te_geo_hour'].fillna(global_mean_48).values

test_mapped_ts = test.merge(full_series_ts.rename('te_geo_ts').reset_index(), on=['geohash', 'time_slot'], how='left')
test['te_geo_ts'] = test_mapped_ts['te_geo_ts'].fillna(global_mean_48).values

# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================
def engineer_master_features(df):
    df = df.copy()

    # Day of Week Logic
    if 'day' in df.columns:
        df['DayOfWeek'] = df['day'] % 7
        df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)

    # Fill Missing Values
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    df['Weather'] = df['Weather'].fillna('Unknown')
    df['geohash'] = df['geohash'].fillna('Unknown')
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['NumberofLanes'] = df['NumberofLanes'].fillna(df['NumberofLanes'].median())

    # Binary Mappings
    df['LargeVehicles'] = df['LargeVehicles'].map({'Allowed': 1, 'Not Allowed': 0}).fillna(0)
    df['Landmarks'] = df['Landmarks'].map({'Yes': 1, 'No': 0}).fillna(0)

    # Advanced Feature Interactions
    df['Road_Lanes'] = df['RoadType'].astype(str) + "_" + df['NumberofLanes'].astype(str)
    df['geo_zone_4'] = df['geohash'].astype(str).str[:4]
    df['geo_zone_5'] = df['geohash'].astype(str).str[:5]
    df['Geo_Hour'] = df['geohash'].astype(str) + "_Hour" + df['Hour'].astype(str)

    return df

print("Building Advanced Features...")
train_clean = engineer_master_features(train)
test_clean = engineer_master_features(test)

drop_cols = ['Index', 'demand', 'timestamp', 'geohash_mean_48']
X = train_clean.drop(drop_cols, axis=1, errors='ignore')
y = train_clean['demand']
X_test = test_clean.drop(['Index', 'timestamp', 'geohash_mean_48'], axis=1, errors='ignore')[X.columns]

cat_cols = ['geohash', 'RoadType', 'Weather', 'Road_Lanes', 'geo_zone_4', 'geo_zone_5', 'Geo_Hour']

# ==========================================
# 5. TRAINING 10-FOLD CATBOOST (GPU)
# ==========================================
print("\n--- Training 10-Fold CatBoost Regressor (GPU) ---")
X_cat = X.copy()
X_test_cat = X_test.copy()
for col in cat_cols:
    X_cat[col] = X_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

cat_test_predictions = np.zeros(len(X_test))
cat_oof_predictions = np.zeros(len(X))

fold = 1
for train_idx, val_idx in kf.split(X_cat):
    print(f"   -> Fold {fold}...")
    X_tr, X_va = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.04,
        depth=8,
        cat_features=cat_cols,
        eval_metric='RMSE',
        random_seed=42+fold,
        early_stopping_rounds=150,
        task_type='GPU',
        verbose=1000
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

    cat_oof_predictions[val_idx] = model.predict(X_va)
    cat_test_predictions += model.predict(X_test_cat) / 10
    fold += 1

# ==========================================
# 6. TRAINING 10-FOLD XGBOOST (CUDA)
# ==========================================
print("\n--- Training 10-Fold XGBoost Regressor (CUDA) ---")
X_xgb = X.copy()
X_test_xgb = X_test.copy()
for col in cat_cols:
    X_xgb[col] = X_xgb[col].astype('category')
    X_test_xgb[col] = X_test_xgb[col].astype('category')

xgb_test_predictions = np.zeros(len(X_test))
xgb_oof_predictions = np.zeros(len(X))

fold = 1
for train_idx, val_idx in kf.split(X_xgb):
    print(f"   -> Fold {fold}...")
    X_tr, X_va = X_xgb.iloc[train_idx], X_xgb.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=2.0,
        random_state=42+fold,
        tree_method='hist',
        device='cuda',
        enable_categorical=True,
        verbosity=0
    )
    model.set_params(early_stopping_rounds=150)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )

    xgb_oof_predictions[val_idx] = model.predict(X_va)
    xgb_test_predictions += model.predict(X_test_xgb) / 10
    fold += 1

# ==========================================
# 7. ENSEMBLE BLENDING & CLIPPING
# ==========================================
print("\nBlending CatBoost and XGBoost predictions...")
# Optimal weights: 15% CatBoost, 85% XGBoost
w_cat = 0.15
w_xgb = 0.85

oof_final = w_cat * cat_oof_predictions + w_xgb * xgb_oof_predictions
test_final = w_cat * cat_test_predictions + w_xgb * xgb_test_predictions

oof_final = np.clip(oof_final, a_min=0, a_max=1)
test_final = np.clip(test_final, a_min=0, a_max=1)

# Scores
global_r2 = r2_score(y, oof_final)
day49_mask = (train_clean['day'] == 49)
day49_r2 = r2_score(y[day49_mask], oof_final[day49_mask])

print("\n" + "*"*30)
print(f"GLOBAL OOF R2 (Unleaked): {global_r2:.5f}")
print(f"DAY 49 OOF R2 (Unleaked): {day49_r2:.5f}")
print(f"ESTIMATED HACKATHON SCORE: {max(0, 100*day49_r2):.2f} / 100")
print("*"*30 + "\n")

submission = pd.DataFrame({
    'Index': test_index,
    'demand': test_final
})

sub_path = os.path.join(OUTPUT_DIR, 'T11_submission.csv')
submission.to_csv(sub_path, index=False)
print(f"Saved to '{sub_path}'!")

submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_ultimate.csv'), index=False)
print("Saved a copy to 'submission_ultimate.csv'!")

# Zip files
print("Zipping source files...")
zip_path = os.path.join(OUTPUT_DIR, 'T11_source_files.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('train_cuda_ensemble_final.py', arcname='train_cuda_ensemble_final.py')
print(f"Saved source archive to '{zip_path}'!")
