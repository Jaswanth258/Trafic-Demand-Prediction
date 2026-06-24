import pandas as pd
import numpy as np
import os
import zipfile
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

print("RUNNING ULTIMATE CATBOOST + LIGHTGBM ENSEMBLE PIPELINE FOR T9 SUBMISSION...")

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
# 3. FEATURE ENGINEERING
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
# 4. TRAINING 10-FOLD CATBOOST
# ==========================================
print("\n--- Training 10-Fold CatBoost Regressor ---")
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
        eval_metric='R2',
        random_seed=42+fold,
        early_stopping_rounds=150,
        thread_count=-1,
        verbose=1000
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

    cat_oof_predictions[val_idx] = model.predict(X_va)
    cat_test_predictions += model.predict(X_test_cat) / 10
    fold += 1

# ==========================================
# 5. TRAINING 10-FOLD LIGHTGBM
# ==========================================
print("\n--- Training 10-Fold LightGBM Regressor ---")
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

lgb_test_predictions = np.zeros(len(X_test))
lgb_oof_predictions = np.zeros(len(X))

fold = 1
for train_idx, val_idx in kf.split(X_lgb):
    print(f"   -> Fold {fold}...")
    X_tr, X_va = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=127,
        feature_fraction=0.7,
        bagging_fraction=0.75,
        bagging_freq=5,
        min_child_samples=25,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42+fold,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(1000)]
    )

    lgb_oof_predictions[val_idx] = model.predict(X_va)
    lgb_test_predictions += model.predict(X_test_lgb) / 10
    fold += 1

# ==========================================
# 6. ENSEMBLE BLENDING & CLIPPING
# ==========================================
print("\nBlending CatBoost and LightGBM predictions...")
# Optimal weights: 30% CatBoost, 70% LightGBM
w_cat = 0.30
w_lgb = 0.70

oof_final = w_cat * cat_oof_predictions + w_lgb * lgb_oof_predictions
test_final = w_cat * cat_test_predictions + w_lgb * lgb_test_predictions

oof_final = np.clip(oof_final, a_min=0, a_max=1)
test_final = np.clip(test_final, a_min=0, a_max=1)

# Scores
global_r2 = r2_score(y, oof_final)
day49_mask = (train_clean['day'] == 49)
day49_r2 = r2_score(y[day49_mask], oof_final[day49_mask])

print("\n" + "*"*30)
print(f"GLOBAL OOF R2: {global_r2:.5f}")
print(f"DAY 49 OOF R2: {day49_r2:.5f}")
print(f"ESTIMATED HACKATHON SCORE: {max(0, 100*day49_r2):.2f} / 100")
print("*"*30 + "\n")

submission = pd.DataFrame({
    'Index': test_index,
    'demand': test_final
})

sub_path = os.path.join(OUTPUT_DIR, 'T9_submission.csv')
submission.to_csv(sub_path, index=False)
print(f"Saved to '{sub_path}'!")

submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_ultimate.csv'), index=False)
print("Saved a copy to 'submission_ultimate.csv'!")

# Zip files
print("Zipping source files...")
zip_path = os.path.join(OUTPUT_DIR, 'T9_source_files.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('train_ensemble_final.py', arcname='train_ensemble_final.py')
print(f"Saved source archive to '{zip_path}'!")
