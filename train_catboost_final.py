import pandas as pd
import numpy as np
import os
import zipfile
import warnings
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

print("RUNNING ULTIMATE CATBOOST PIPELINE FOR T8 SUBMISSION...")

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

    # Convert all categoricals to strings for CatBoost
    cat_cols = ['geohash', 'RoadType', 'Weather', 'Road_Lanes', 'geo_zone_4', 'geo_zone_5', 'Geo_Hour']
    for col in cat_cols:
        df[col] = df[col].astype(str)

    return df, cat_cols

print("Building Advanced Features...")
train_clean, cat_features = engineer_master_features(train)
test_clean, _ = engineer_master_features(test)

# Drop raw timestamp, Index, demand, and temp helpers
drop_cols = ['Index', 'demand', 'timestamp', 'geohash_mean_48']
X = train_clean.drop(drop_cols, axis=1, errors='ignore')
y = train_clean['demand']
X_test = test_clean.drop(['Index', 'timestamp', 'geohash_mean_48'], axis=1, errors='ignore')[X.columns]

# ==========================================
# 4. 10-FOLD CATBOOST (Maximum Accuracy)
# ==========================================
print("Training 10 Models to find the absolute best predictions...")

kf = KFold(n_splits=10, shuffle=True, random_state=42)

test_predictions = np.zeros(len(X_test))
oof_predictions = np.zeros(len(X))

fold = 1
for train_idx, val_idx in kf.split(X):
    print(f"\n   -> Training Model {fold} of 10...")
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
        iterations=2500,          # Massive number of trees
        learning_rate=0.04,       # Very precise learning rate
        depth=8,                  # Deep enough to catch complex traffic patterns
        cat_features=cat_features,
        eval_metric='R2',         # Directly optimizes for Hackathon metric
        random_seed=42+fold,
        early_stopping_rounds=150,# Prevent overfitting
        thread_count=-1,          # Use all CPU cores
        verbose=500               # Log every 500 trees to show progress
    )

    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True
    )

    oof_predictions[val_idx] = model.predict(X_va)
    test_predictions += model.predict(X_test) / 10  # Average out 10 predictions
    fold += 1

# Clip predictions
oof_predictions = np.clip(oof_predictions, a_min=0, a_max=1)
test_predictions = np.clip(test_predictions, a_min=0, a_max=1)

# ==========================================
# 5. FINAL SCORE & SUBMISSION
# ==========================================
final_r2 = r2_score(y, oof_predictions)
competition_score = max(8, 100 * final_r2)

day49_mask = (train_clean['day'] == 49)
day49_r2 = r2_score(y[day49_mask], oof_predictions[day49_mask])
day49_score = max(0, 100 * day49_r2)

print("\n" + "*"*15)
print(f"FINAL LOCAL R2 SCORE (GLOBAL): {final_r2:.5f}")
print(f"FINAL LOCAL R2 SCORE (DAY 49 ONLY): {day49_r2:.5f}")
print(f"ESTIMATED HACKATHON SCORE: {day49_score:.2f} / 100")
print("*"*15 + "\n")

submission = pd.DataFrame({
    'Index': test_index,
    'demand': test_predictions
})

sub_path = os.path.join(OUTPUT_DIR, 'T8_submission.csv')
submission.to_csv(sub_path, index=False)
print(f"Saved to '{sub_path}'!")

# Also write a copy to submission_ultimate.csv
submission.to_csv(os.path.join(OUTPUT_DIR, 'submission_ultimate.csv'), index=False)
print("Saved a copy to 'submission_ultimate.csv'!")

# ==========================================
# 6. ZIP SOURCE FILES FOR UPLOAD
# ==========================================
print("Zipping source files...")
zip_path = os.path.join(OUTPUT_DIR, 'T8_source_files.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write('train_catboost_final.py', arcname='train_catboost_final.py')
print(f"Saved source archive to '{zip_path}'!")
