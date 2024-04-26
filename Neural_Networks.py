import dataset
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics

def run():
    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = dataset.get_data()

    # Create a Sequential model
    model = Sequential()

    # Add an input layer and a hidden layer with 20 neurons
    model.add(Dense(30, input_dim=X_train.shape[1], activation='sigmoid'))

    # Add a output layer with 1 neuron
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 訓練模型
    model.fit(X_train, y_train, epochs=50, batch_size=50)

    # 預測測試集結果
    y_pred = model.predict(X_test)

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