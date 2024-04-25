import dataset
from sklearn.svm import SVR
from sklearn import metrics

# 劃分訓練集和測試集
X_train, X_test, y_train, y_test = dataset.get_data()

# 創建支援向量機迴歸器對象
regressor = SVR(
    kernel='rbf', # 核函數
    degree=3, # 多項式核函數的次數
    gamma='scale', # 核函數的係數
    coef0=0.0, # 核函數的獨立項
    tol=0.001, # 容忍度
    C=1.0, # 懲罰係數
    epsilon=0.1, # epsilon-tube
    shrinking=True, # 是否使用收縮啟發式算法
    cache_size=200, # 核緩存大小
    verbose=False, # 是否輸出日誌
    max_iter=-1 # 最大迭代次數
)

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