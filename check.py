import dataset
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split

# 加載數據集
X, y = dataset.load_data()

print(type(X))

# print(X.isnull().sum())
print(X)


# print(X.isnull.sum())
# X.isnull.sum()

# 劃分訓練集和測試集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

