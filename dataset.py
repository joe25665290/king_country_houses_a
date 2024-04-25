import pandas as pd
from sklearn.model_selection import train_test_split
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

def data_split(X, y):
    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # save to csv
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

if __name__ == '__main__':
    X, y = load_data()
    data_split(X, y)
    print('Done!')

def get_data():
    # return X_train, X_test, y_train, y_test
    X_train = pd.read_csv('X_train.csv')
    X_test = pd.read_csv('X_test.csv')
    y_train = pd.read_csv('y_train.csv')
    y_test = pd.read_csv('y_test.csv')
    return X_train, X_test, y_train, y_test