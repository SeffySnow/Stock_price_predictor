

def preprocess_stock_data(df):
    # 1. Data Cleaning
    
    print("Missing values:\n", df.isnull().sum())

    # Fill missing values using forward fill (common for stock data)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')  

    # 2. Feature Engineering
    # Create technical indicators and derived features
    processed_df = df.copy()

    # Moving Averages (5-day and 20-day)
    processed_df['MA5'] = processed_df['Close'].rolling(window=5).mean()
    processed_df['MA20'] = processed_df['Close'].rolling(window=20).mean()

    # Price differences
    processed_df['Price_Diff'] = processed_df['Close'].diff()
    processed_df['Open_Close_Diff'] = processed_df['Close'] - processed_df['Open']

    # Daily returns
    processed_df['Daily_Return'] = processed_df['Close'].pct_change()

    # Volatility (rolling 20-day standard deviation)
    processed_df['Volatility'] = processed_df['Daily_Return'].rolling(window=20).std()

    # Relative Strength Index (RSI)
    def calculate_rsi(data, periods=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    processed_df['RSI'] = calculate_rsi(processed_df['Close'])

    # Volume-based features
    processed_df['Volume_MA5'] = processed_df['Volume'].rolling(window=5).mean()
    processed_df['Volume_Change'] = processed_df['Volume'].pct_change()

    # 3. Create target variable (next day's closing price)
    processed_df['Target'] = processed_df['Close'].shift(-1)

    # 4. Remove rows with NaN values created by rolling calculations
    processed_df = processed_df.dropna()
    features = ['Close', 'High', 'Low', 'Open', 'Volume',
            'MA5', 'MA20', 'Price_Diff', 'Open_Close_Diff',
            'Daily_Return', 'Volatility', 'RSI',
            'Volume_MA5', 'Volume_Change']
    X = processed_df[features]
    y = processed_df['Target']

    return X,y



