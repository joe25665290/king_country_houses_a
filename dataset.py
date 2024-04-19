import pandas as pd

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
        'sqft_above',
        'sqft_basement',
        'yr_built',
        'yr_renovated',
        'lat',
        'long',
        'sqft_living15',
        'sqft_lot15']]
    target = df['price']
    
    return data, target