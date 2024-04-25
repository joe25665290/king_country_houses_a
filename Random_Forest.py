import dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# 劃分訓練集和測試集
X_train, X_test, y_train, y_test = dataset.get_data()

# 創建隨機森林迴歸器對象
regressor = RandomForestRegressor(n_estimators=100)

# 訓練模型
regressor = regressor.fit(X_train,y_train)

# 預測測試集結果
y_pred = regressor.predict(X_test)

# 計算模型的 R^2 分數
print("R^2 Score:",metrics.r2_score(y_test, y_pred))

# 計算模型的均方誤差
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
# 計算模型的平均絕對誤差
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))