import pandas as pd
# from sklearn.preprocessing import StandardScaler

def z_score_normalization(df, cols):
    df_normalized = df.copy()
    for col in cols:
        all_col_data = df_normalized[col].copy()
        # print(all_col_data)
        mu = all_col_data.mean()
        std = all_col_data.std()
        
        z_score_normalized = (all_col_data - mu) / std
        df_normalized[col] = z_score_normalized
    return df_normalized

def load_data():
    # 讀取 CSV 檔案
    df = pd.read_csv('king_ country_ houses_aa.csv')

    # 將 'X' 和 'y' 設定為你的特徵和目標變量
    data = df[[
        'bedrooms',
        'bathrooms',
        'sqft_living',
        'sqft_lot',
        'floors',
        'waterfront',
        'view',
        'condition',
        'grade',
        'sqft_basement',
        'yr_built',
        'yr_renovated',
        'lat',
        'long',]]
    target = df['price']

    normalized = pd.DataFrame(z_score_normalization(data, ['yr_built', 'yr_renovated', 'lat', 'long']))
    
    data.drop(['yr_built', 'yr_renovated', 'lat', 'long'], axis=1)

    data = pd.merge(data, normalized, left_index=True, right_index=True)

    return data, target