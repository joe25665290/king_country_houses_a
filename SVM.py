import dataset

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics

# 加載數據集
X, y = dataset.load_data()

# 劃分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 

# 創建支援向量機迴歸器對象
regressor = SVR()

# 訓練模型
regressor = regressor.fit(X_train,y_train)

# 預測測試集結果
y_pred = regressor.predict(X_test)

# 計算模型的 R^2 分數
print("R^2 Score:",metrics.r2_score(y_test, y_pred))

# 計算模型的均方誤差
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))