import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# Load data
train = pd.read_csv('train (1).csv')
test = pd.read_csv('test (1).csv')

def prepare_data(df):
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df = df.ffill().bfill()
    
    ndvi_cols = ['ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw']
    df['ndvi_smooth'] = df[ndvi_cols].mean(axis=1).rolling(window=4, min_periods=1).mean()
    
    df['reanalysis_specific_humidity_g_per_kg'] = df.groupby('city')['reanalysis_specific_humidity_g_per_kg'].transform(
        lambda x: x.rolling(window=4, center=True, min_periods=1).median()
    )
    
    df['week_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 53)
    df['week_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 53)
    
    main_cols = ['reanalysis_specific_humidity_g_per_kg', 'station_min_temp_c', 'reanalysis_dew_point_temp_k']
    for col in main_cols:
        for lag in [4, 8, 12, 16, 24]:
            df[f'{col}_lag_{lag}'] = df.groupby('city')[col].shift(lag)
        df[f'{col}_delta'] = df[col] - df.groupby('city')[col].shift(4)
            
    return df.ffill().bfill()

train_eng = prepare_data(train)
test_eng = prepare_data(test)

all_results = []

for city in ['sj', 'iq']:
    tr_c = train_eng[train_eng['city'] == city].copy()
    te_c = test_eng[test_eng['city'] == city].copy()
    
    week_means = tr_c.groupby('weekofyear')['total_cases'].mean()
    tr_c['week_baseline'] = tr_c['weekofyear'].map(week_means)
    te_c['week_baseline'] = te_c['weekofyear'].map(week_means)
    
    exclude = ['id', 'city', 'year', 'week_start_date', 'total_cases']
    feats = [c for c in tr_c.columns if c not in exclude]
    X, y = tr_c[feats], tr_c['total_cases']
    
    m1 = HistGradientBoostingRegressor(loss='poisson', learning_rate=0.01, max_iter=315, random_state=0).fit(X, y)
    m2 = HistGradientBoostingRegressor(loss='absolute_error', learning_rate=0.01, max_iter=315, random_state=0).fit(X, y)
    m3 = RandomForestRegressor(n_estimators=95, max_depth=4, random_state=0).fit(X, y)
    
    preds = 0.04 * m1.predict(te_c[feats]) + 0.80 * m2.predict(te_c[feats]) + 0.04 * m3.predict(te_c[feats])
    
    out = pd.DataFrame({'id': te_c['id'], 'total_cases': np.round(preds).astype(int)})
    all_results.append(out)

submission = pd.concat(all_results).sort_values('id')
submission['total_cases'] = submission['total_cases'].clip(lower=0).astype(int)
submission.to_csv('submission.csv', index=False)

print("Triple-Ensemble Model Complete!")
print(f"Prediction Average: {submission['total_cases'].mean():.2f}")
print(f"Prediction Max: {submission['total_cases'].max()}")