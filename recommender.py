def ensemble_trading_recommendation(current_price, y_pred_lr, y_pred_lstm, y_pred_xgb, y_pred_svr, threshold=0.02):
    """
    Generate a trading recommendation based on the ensemble of model predictions.

    Parameters:
        current_price (float): The current stock price.
        y_pred_lr, y_pred_lstm, y_pred_xgb, y_pred_svr (np.array or float): 
            The predicted stock prices from Linear Regression, LSTM, XGBoost, and SVR models.
            (If arrays, you can use the last value as the most recent prediction.)
        threshold (float): The minimum percentage change to trigger a recommendation 
                           (e.g., 0.02 for 2%).

    Returns:
        str: "Buy", "Sell", or "Hold" recommendation.
    """
    # If predictions are arrays, take the last predicted value (most recent forecast)
    pred_lr = y_pred_lr[-1] if hasattr(y_pred_lr, '__len__') else y_pred_lr
    pred_lstm = y_pred_lstm[-1] if hasattr(y_pred_lstm, '__len__') else y_pred_lstm
    pred_xgb = y_pred_xgb[-1] if hasattr(y_pred_xgb, '__len__') else y_pred_xgb
    pred_svr = y_pred_svr[-1] if hasattr(y_pred_svr, '__len__') else y_pred_svr
    
    # Compute the ensemble prediction (simple average)
    ensemble_pred = (pred_lr + pred_lstm + pred_xgb + pred_svr) / 4.0
    print("Ensemble predicted price:", ensemble_pred)
    
    # Compare ensemble prediction to current price using the threshold
    if ensemble_pred >= current_price * (1 + threshold):
        return "Buy"
    elif ensemble_pred <= current_price * (1 - threshold):
        return "Sell"
    else:
        return "Hold"
