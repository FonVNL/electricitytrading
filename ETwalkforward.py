import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

# =========================
# 1. Load CSV
# =========================
df = pd.read_csv("Netherlands.csv", parse_dates=["Datetime (Local)"])
df = df.sort_values("Datetime (Local)").set_index("Datetime (Local)")
df = df[["Price (EUR/MWhe)"]].rename(columns={"Price (EUR/MWhe)": "price"})

# =========================
# 2. Feature engineering (UPGRADED)
# =========================
# Basic price features
df['price_lag_1'] = df['price'].shift(1)
df['price_lag_2'] = df['price'].shift(2)
df['price_change_1'] = df['price'].diff(1)
df['price_change_2'] = df['price'].diff(2)

# Mean reversion
df['price_vs_daily_avg'] = df['price'] - df['price'].rolling(24).mean()

rolling_mean_24 = df['price'].rolling(24).mean()
rolling_std_24 = df['price'].rolling(24).std()
df['zscore'] = (df['price'] - rolling_mean_24) / rolling_std_24

df['price_prev_day'] = df['price'].shift(24)
df['delta_prev_day'] = df['price'] - df['price_prev_day']

# Momentum
df['momentum_3'] = df['price'] - df['price'].shift(3)
df['momentum_12'] = df['price'] - df['price'].shift(12)

# Volatility
df['volatility_6'] = df['price'].rolling(6).std()
df['volatility_24'] = df['price'].rolling(24).std()

# Time features (cyclical)
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
df['dayofweek'] = df.index.dayofweek

# Target
df['target'] = df['price'].shift(-1) - df['price']

# Drop NaNs
df = df.dropna()

# =========================
# 3. Feature list
# =========================
features = [
    'price_lag_1', 'price_lag_2',
    'price_change_1', 'price_change_2',

    'price_vs_daily_avg',
    'zscore',
    'delta_prev_day',

    'momentum_3',
    'momentum_12',

    'volatility_6',
    'volatility_24',

    'hour_sin',
    'hour_cos',
    'dayofweek'
]

# =========================
# 4. Walk-forward setup
# =========================
window = 28 * 24   # 1 year
step = 7 * 24       # weekly retrain

all_results = []

# =========================
# 5. Walk-forward loop
# =========================
for start in range(window, len(df) - step, step):

    train = df.iloc[start - window:start]
    test = df.iloc[start:start + step].copy()

    if len(train) == 0 or len(test) == 0:
        continue

    X_train, y_train = train[features], train['target']
    X_test, y_test = test[features], test['target']

    # Model (balanced speed + performance)
    model = LGBMRegressor(
        n_estimators=150,
        max_depth=-1,
        learning_rate=0.05,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    pred_change = model.predict(X_test)

    # =========================
    # 6. Signal (IMPROVED)
    # =========================
    # Volatility-adjusted threshold
    vol = test['volatility_24']
    dynamic_threshold = 0.5 * vol

    signal = np.where(
        np.abs(pred_change) > dynamic_threshold,
        np.clip(pred_change / (vol + 1e-6), -1, 1),
        0
    )

    # =========================
    # 7. PnL
    # =========================
    test['pred'] = pred_change
    test['signal'] = signal
    test['real_move'] = test['price'].shift(-1) - test['price']

    cost = 0.05
    test['pnl'] = test['signal'] * test['real_move'] - np.abs(test['signal']) * cost

    test = test.dropna()

    all_results.append(test)

# =========================
# 8. Combine results
# =========================
if len(all_results) == 0:
    raise ValueError("Not enough data for 1-year training window.")

results = pd.concat(all_results)

# =========================
# 9. Metrics
# =========================
pred_direction = np.sign(results['pred'])
true_direction = np.sign(results['target'])

accuracy = (pred_direction == true_direction).mean()
total_pnl = results['pnl'].sum()

trade_mask = results['signal'] != 0

hit_rate = (
    np.sign(results.loc[trade_mask, 'pnl']) ==
    np.sign(results.loc[trade_mask, 'real_move'])
).mean()

num_trades = trade_mask.sum()

# =========================
# 10. Cumulative PnL
# =========================
results['cum_pnl'] = results['pnl'].cumsum()

# =========================
# 11. Daily PnL
# =========================
daily_pnl = results['pnl'].resample('D').sum()
daily_pnl_df = daily_pnl.to_frame(name='pnl')
daily_pnl_df['year'] = daily_pnl_df.index.year

# =========================
# 12. Plot per year
# =========================
for y in daily_pnl_df['year'].unique():
    yearly = daily_pnl_df[daily_pnl_df['year'] == y]

    plt.figure(figsize=(12, 5))

    cum_year = yearly['pnl'].cumsum()
    colors = ['green' if x >= 0 else 'red' for x in yearly['pnl']]

    plt.plot(yearly.index, cum_year, label='Cumulative PnL')
    plt.bar(yearly.index, yearly['pnl'], alpha=0.4, color=colors, label='Daily PnL')

    plt.title(f"Walk-forward PnL - {y}")
    plt.xlabel("Date")
    plt.ylabel("PnL (€)")
    plt.legend()
    plt.show()

# =========================
# 13. Print results
# =========================
print(f"Accuracy: {accuracy:.4f}")
print(f"Total PnL: {total_pnl:.2f}")
print(f"Hit rate: {hit_rate:.4f}")
print(f"Number of trades: {num_trades}")