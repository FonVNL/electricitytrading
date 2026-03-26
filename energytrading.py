import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# =========================
# 1. Load CSV
# =========================
df = pd.read_csv("Netherlands.csv", parse_dates=["Datetime (Local)"])
df = df.sort_values("Datetime (Local)").set_index("Datetime (Local)")
df = df[["Price (EUR/MWhe)"]].rename(columns={"Price (EUR/MWhe)": "price"})

# =========================
# 2. Feature engineering (price only)
# =========================
df['price_lag_1'] = df['price'].shift(1)
df['price_lag_2'] = df['price'].shift(2)
df['price_change_1'] = df['price'].diff(1)
df['price_change_2'] = df['price'].diff(2)
df['rolling_mean_3'] = df['price'].rolling(3).mean()
df['rolling_std_3'] = df['price'].rolling(3).std()

# Time features
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

# =========================
# 3. Target: next-hour price change
# =========================
df['target'] = df['price'].shift(-1) - df['price']
df = df.dropna()

# =========================
# 4. Train/test split
# =========================
split = int(len(df) * 0.7)
train = df.iloc[:split]
test = df.iloc[split:]

features = [
    'price_lag_1', 'price_lag_2',
    'price_change_1', 'price_change_2',
    'rolling_mean_3', 'rolling_std_3',
    'hour', 'dayofweek'
]

X_train, y_train = train[features], train['target']
X_test, y_test = test[features], test['target']

# =========================
# 5. Train regression model
# =========================
model = LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
model.fit(X_train, y_train)

# =========================
# 6. Predict next-hour price change
# =========================
pred_change_predicted = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_change_predicted)

# Accuracy (direction)
pred_direction = np.sign(pred_change_predicted)
true_direction = np.sign(y_test)
accuracy = (pred_direction == true_direction).mean()

# =========================
# 7. Trading signals with position sizing
# =========================
test = test.copy()
test['pred_change'] = pred_change_predicted
# Position size proportional to expected move, clipped
test['signal'] = np.clip(test['pred_change'], -1, 1)

# =========================
# 8. PnL simulation
# =========================
test['real_move'] = test['price'].shift(-1) - test['price']
cost = 0.05  # transaction cost per trade
test['pnl'] = test['signal'] * test['real_move'] - abs(test['signal']) * cost
test['cum_pnl'] = test['pnl'].cumsum()

# =========================
# 9. Daily PnL
# =========================
daily_pnl = test['pnl'].resample('D').sum()
daily_pnl_df = daily_pnl.to_frame(name='pnl')
daily_pnl_df['year'] = daily_pnl_df.index.year

# =========================
# 10. Plot daily + cumulative PnL per year
# =========================
years = daily_pnl_df['year'].unique()
for y in years:
    plt.figure(figsize=(12, 5))
    yearly_pnl = daily_pnl_df[daily_pnl_df['year'] == y]

    # Cumulative PnL for the year
    cum_pnl_year = yearly_pnl['pnl'].cumsum()

    # Color bars: green for profit, red for loss
    colors = ['green' if x >= 0 else 'red' for x in yearly_pnl['pnl'].values]

    plt.plot(cum_pnl_year, label='Cumulative PnL', color='blue')
    plt.bar(yearly_pnl.index, yearly_pnl['pnl'], alpha=0.4, color=colors, label='Daily PnL')
    plt.title(f"Daily and Cumulative PnL - {y}")
    plt.xlabel("Date")
    plt.ylabel("Profit (€)")
    plt.legend()
    plt.show()

# =========================
# 11. Print key metrics
# =========================
total_pnl = test['pnl'].sum()
hit_rate = (np.sign(test['pnl']) == np.sign(test['real_move'])).mean()
num_trades = (test['signal'].abs() > 0).sum()

print(f"Accuracy: {accuracy:.4f}")
print(f"Total PnL: {total_pnl:.2f}")
print(f"Hit rate: {hit_rate:.4f}")
print(f"Number of trades: {num_trades}")