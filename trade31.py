import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib

df = pd.read_csv('price-data.csv', index_col='Date', parse_dates=True)

df['SMA'] = df['Close'].rolling(window=10).mean()
df['EMA'] = df['Close'].ewm(span=10, adjust=False).mean()
df['RSI'] = 100 - (
    100 / (1 + (df['Close'].diff() < 0).rolling(window=14).sum() /
          (df['Close'].diff() > 0).rolling(window=14).sum()))
df['MACD'] = (df['Close'].ewm(span=12, adjust=False).mean() -
              df['Close'].ewm(span=26, adjust=False).mean())
df['y_price'] = df['Close'].shift(-1)
df['y_price_change'] = df['Close'].shift(-1) - df['Close']
df['y_flip'] = np.where(df['Close'].shift() - df['Close'] > 0, 1, 0)
df['y_return'] = np.log(df['Close'].shift(-1) / df['Close'])
df['y'] = np.where(df['y_price_change'] > 0, 1, 0)

# データの欠損値を平均値で埋める
df.fillna(df.mean(), inplace=True)

print("末尾のデータ:\n", df.tail())

# Data normalization
X = df.drop(['y'], axis=1)
y = df['y']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'max_depth': 10,
    'n_estimators': 900,
    'verbose': -1,
}

model = LGBMClassifier(**params)

# Train the model
model.fit(X_train, y_train)

# Calculate AUC score for prediction accuracy
auc_train = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
auc_test = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print("Training data AUC:", auc_train)
print("Test data AUC:", auc_test)

current_price = df['Close'].iloc[-1]
print("Latest current price:", current_price)

# Calculate current volatility
volatility = df["Close"].rolling(window=20).std().iloc[-1]
print("Current volatility:", volatility)

# Predict the signal for the latest data
predicted_signal = model.predict(X.iloc[-1].values.reshape(1, -1))

signal = "BUY" if predicted_signal[0] == 1 else "SELL"

print("売買シグナル:", signal)

if volatility < 5.8:
    signal = "CLOSE"

print("現在の売買シグナル:", signal)

# シグナルをファイルに保存する
signal_file = r'D:\Downloads\Bybit41\MQL4\Files\signal.txt'
with open(signal_file, "w") as f:
    f.write(signal)
