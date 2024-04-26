import data_process.dataset as dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def run():
    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = dataset.get_data()

    # Create a Polynomial Features object
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False, order='C')

    # Transform the X_train and X_test to include polynomial features
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Create a Linear Regression model
    regressor = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)

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