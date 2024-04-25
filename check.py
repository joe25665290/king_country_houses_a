import dataset
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split

# 加載數據集
train_Data, train_target, test_Data, test_target = dataset.load_data()

print(train_Data)
print(train_target)
print(test_Data)
print(test_target)

# print(X.isnull().sum())
# print(X)


# print(X.isnull.sum())
# X.isnull.sum()

# 劃分訓練集和測試集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 

