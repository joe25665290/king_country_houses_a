import data_process.dataset as dataset
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

def run():

    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = dataset.get_data()

    # 創建決策樹分類器對象
    regressor = DecisionTreeRegressor( random_state=0, 
                                    max_depth=150, 
                                    min_samples_split=20,    
                                    min_samples_leaf=5,
                                    max_features=10,
                                    max_leaf_nodes=200,
                                    min_impurity_decrease=0.1,
                                    splitter='best')

    # 訓練模型
    regressor = regressor.fit(X_train,y_train)

    # 預測測試集結果
    y_pred = regressor.predict(X_test)

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