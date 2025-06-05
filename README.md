# 🚀 Tesla Stock Price Prediction Project

_A data-driven approach to forecast Tesla’s next-day closing price using technical indicators and ensemble modeling for trading recommendations._

---

## 🎯 Approach Overview
1. **Feature Engineering**  
   - Started by creating new technical indicators from raw price and volume data.  
   - Dropped any rows with missing values to ensure a clean dataset.  
   - Selected the following features to capture influential signals:
     ```
     ['Close', 'High', 'Low', 'Open', 'Volume',
      'MA5', 'MA20', 'Price_Diff', 'Open_Close_Diff',
      'Daily_Return', 'Volatility', 'RSI',
      'Volume_MA5', 'Volume_Change']
     ```
   - Defined the target variable (`Target`) as the next-day closing price.

2. **Model Training & Evaluation**  
   - Trained four different regression models on the engineered features:
     - **Linear Regression (LR)**
     - **Long Short-Term Memory (LSTM)**
     - **XGBoost**
     - **Support Vector Regression (SVR)**
   - Evaluated each model using standard metrics: MAE, MSE, RMSE, and R-squared.

3. **Ensemble Trading Recommendation**  
   - After identifying the best-performing model, combined predictions from all four models into a simple average ensemble.  
   - The ensemble output is compared against the current price with a configurable threshold (2% by default).  
   - Based on this comparison, the function outputs one of three signals: **Buy**, **Sell**, or **Hold**.

---

## 📊 Model Performance

| Model      | MAE     | MSE      | RMSE    | R-squared |
|:----------:|:-------:|:--------:|:-------:|:---------:|
| **LR**       | 5.1822  | 49.5127  | 7.0365  | 0.9587    |
| **LSTM**     | 10.1761 | 179.2666 | 13.3890 | 0.8495    |
| **XGBoost**  | 6.2216  | 68.6233  | 8.2839  | 0.9428    |
| **SVR**      | 7.1496  | 96.4042  | 9.8186  | 0.9197    |

---

## 🧩 Ensemble Recommendation Logic
- **Inputs**:  
  - Current stock price (`current_price`)  
  - Last predictions from each model (`y_pred_lr`, `y_pred_lstm`, `y_pred_xgb`, `y_pred_svr`)  
  - Threshold parameter (default 0.02 for 2%)  

- **How It Works**:  
  1. Extract the most recent prediction from each model.  
  2. Compute the ensemble prediction as the **average** of the four model outputs.  
  3. Compare the ensemble prediction against the current price:
     - If `ensemble_pred ≥ current_price × (1 + threshold)`, signal **“Buy”**.  
     - If `ensemble_pred ≤ current_price × (1 − threshold)`, signal **“Sell”**.  
     - Otherwise, signal **“Hold”**.

This simple ensemble approach smooths out individual model noise and issues a clear trading recommendation based on whether the consensus forecast deviates meaningfully (±2%) from today’s price.

---

## 📁 Output Files
- Model metrics and prediction results are saved in the **`results/`** folder.  
- Fitting and performance plots for each model can be found in the **`plots/`** folder.

---
