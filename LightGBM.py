import data_process.dataset as dataset
import lightgbm as lgb
from sklearn import metrics

def run():
    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = dataset.get_data()

    regressor = lgb.LGBMRegressor(num_leaves=50, 
                                learning_rate=0.2, 
                                n_estimators=30, 
                                random_state=1,   
                                n_jobs=-1)

    # 訓練模型
    regressor = regressor.fit(X_train,y_train)

    # 預測測試集結果
    y_pred = regressor.predict(X_test)

    # 計算模型的 R^2 分數
    # print("R^2 Score:",metrics.r2_score(y_test, y_pred))
    R2 = metrics.r2_score(y_test, y_pred)

    # 計算模型的均方誤差
    # print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
    MSE = metrics.mean_squared_error(y_test, y_pred)

    # 計算模型的平均絕對誤差
    # print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
    MAE = metrics.mean_absolute_error(y_test, y_pred)

    return R2, MSE, MAE

if __name__ == '__main__':
    R2, MSE, MAE = run()
    print(R2)
    print(MSE)
    print(MAE)