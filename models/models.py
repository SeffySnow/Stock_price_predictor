
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def LR(X,y, split_perc = 0.2):
    print("preparing data for linear regression..\n")
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_perc, shuffle=False)
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)
    model = LinearRegression()
    print("Model Training...\n")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # 10. Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"linear regression test results: MAE: {mae:.4} , MSE: {mse:.4}, RMSE: {rmse:.4}, R-squared: {r2:.4} ")
    # 11. Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values(by='coefficient', key=abs, ascending=False))

    plt.figure(figsize=(14, 5))
    plt.plot(y_test.values, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red', alpha=0.7)
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig("plots/Linear_regression.png")
    plt.close()
    return {"MAE":mae,"MSE": mse, "RMSE":rmse, "R-squared":r2}, y_pred, y_test
   



def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def model_LSTM(X, y, time_steps=60, test_size=0.2):
    X = X.values
    y = y.values
    print("preparing data for LSTM..\n")

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    Xs, ys = create_sequences(X_scaled, y_scaled, time_steps)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=test_size, shuffle=False
    )

    model = Sequential([
        LSTM(units=50, return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),

        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error')
    model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    print("Model Training...\n")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    y_pred_scaled = model.predict(X_test)

    # Inverse-transform both predictions and true values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_actual = scaler_y.inverse_transform(y_test)

    # Compute metrics on the same scale
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2  = r2_score(y_test_actual, y_pred)
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    print(f"LSTM test results: MAE: {mae:.4} , MSE: {mse:.4}, RMSE: {rmse:.4}, R-squared: {r2:.4}")

    plt.figure(figsize=(14, 5))
    plt.plot(y_test_actual, color='blue', label='Actual Prices')
    plt.plot(y_pred, color='red', label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig("plots/LSTM.png")
    plt.close()

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R-squared": round(r2, 4)
    }, y_pred



def xgboost_model(X, y, test_size=0.2):
    print("Preparing data for XGBoost...\n")
    # Split the data (no shuffling for time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Optionally scale the features (XGBoost often handles unscaled data, but scaling can help)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define and train the XGBoost regressor
    xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
    print("Training XGBoost model...\n")
    xgb_model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = xgb_model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost test results: MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R-squared: {r2:.4f}")
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.values, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red', alpha=0.7)
    plt.title('XGBoost: Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig("plots/XGBoost.png")
    plt.close()
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R-squared": r2}, y_pred






def svm_model(X, y, test_size=0.2):
    print("Preparing data for SVR...\n")
    # Split the data (without shuffling, to preserve time series order)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Standard scale the features (SVR may benefit from scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define and train the SVR model
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    print("Training SVR model...\n")
    svr_model.fit(X_train_scaled, y_train)
    
    # Predict on test data
    y_pred = svr_model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"SVR test results: MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R-squared: {r2:.4f}")
    
    # Plot Actual vs Predicted Prices
    plt.figure(figsize=(14, 5))
    plt.plot(y_test.values, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red', alpha=0.7)
    plt.title('SVR: Actual vs Predicted Stock Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig("plots/SVR.png")
    plt.close()
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R-squared": r2}, y_pred
