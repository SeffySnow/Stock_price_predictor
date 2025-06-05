

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from preprocessing import preprocess_stock_data
from models import LR ,model_LSTM, xgboost_model,svm_model
from recommender import ensemble_trading_recommendation
import json
import os




def main():
    parser = argparse.ArgumentParser(description="Train the recommendation model")
    parser.add_argument('dataset_name', type=str, nargs='?', default="tsla_data.csv", help="Dataset name (e.g., movie, book)")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    dataset_path = f"dataset/{dataset_name}"
    tsla_data = pd.read_csv(dataset_path)
    tsla_data['Date'] = pd.to_datetime(tsla_data['Date'])
    tsla_data.set_index('Date', inplace=True)
    fig, ax = mpf.plot(tsla_data, type='candle', 
                   title="Tesla (TSLA) Candlestick Chart", 
                   style='yahoo', figsize=(12, 6),
                   returnfig=True)
    fig.savefig("plots/candle_plot.png")

    X, y = preprocess_stock_data(tsla_data)
    
    
    results_lr ,y_pred_lr, y_test= LR(X,y ,0.2)
    results_lstm, y_pred_lstm = model_LSTM (X,y,time_steps = 30, test_size = 0.2)
    results_xgb, y_pred_xgb = xgboost_model(X, y, test_size=0.2)
    results_svr, y_pred_svr = svm_model(X, y, test_size=0.2)

    current_price = y_test[-10]

    recommendation = ensemble_trading_recommendation(
        current_price, y_pred_lr, y_pred_lstm, y_pred_xgb, y_pred_svr, threshold=0.03  # 3% threshold
    )
    print("Trading Recommendation:", recommendation)

    results_file = "results/results.json"

    # Ensure the 'results' directory exists
    os.makedirs("results", exist_ok=True)

    # Load existing results if the file exists, or initialize an empty list if not
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            try:
                results_data = json.load(f)
            except json.decoder.JSONDecodeError:
                results_data = []
    else:
        results_data = []

    # Create a new entry with the model results (assuming results_lr and results_lstm are lists)
    new_entry = {
        "LR": results_lr,
        "LSTM": results_lstm,
         "XGBoost": results_xgb,
         "SVR": results_svr
    }

    # Append the new results to the results list
    results_data.append(new_entry)

    # Write the updated results back to the file
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=4)

    

if __name__ == "__main__":
    main()

