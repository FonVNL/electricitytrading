⚡ Electricity Price Forecasting & Trading Strategy (LightGBM)

This project builds a machine learning-based trading strategy for electricity prices in the Netherlands. It uses historical price data to predict next-hour price changes and simulates a trading strategy based on those predictions.

⸻

🚀 Overview

The script:
	1.	Loads historical electricity price data
	2.	Engineers time-series features
	3.	Trains a LightGBM regression model
	4.	Predicts next-hour price movements
	5.	Generates trading signals
	6.	Simulates profit & loss (PnL)
	7.	Visualizes performance over time

⸻

📊 Key Features
	•	⏱ Time-series forecasting (next-hour price change)
	•	🤖 LightGBM model for regression
	•	📈 Trading strategy simulation
	•	💰 PnL calculation with transaction costs
	•	📅 Daily and cumulative performance plots
	•	🎯 Directional accuracy & hit rate evaluation

⸻

📦 Requirements

Install dependencies using:

pip install pandas numpy matplotlib lightgbm scikit-learn


⸻

📁 Input Data

The script expects a CSV file:

Netherlands.csv

Required columns:
	•	Datetime (Local) → timestamp
	•	Price (EUR/MWhe) → electricity price

⸻

⚙️ How It Works

1. Data Preprocessing
	•	Parses datetime and sets it as index
	•	Sorts data chronologically
	•	Renames price column to price

⸻

2. Feature Engineering

Creates predictive features from historical prices:
	•	Lag features:
	•	price_lag_1, price_lag_2
	•	Price changes:
	•	price_change_1, price_change_2
	•	Rolling statistics:
	•	rolling_mean_3, rolling_std_3
	•	Time features:
	•	hour, dayofweek

⸻

3. Target Variable

The model predicts:

next_hour_price_change = price(t+1) - price(t)


⸻

4. Train/Test Split
	•	70% training data
	•	30% testing data
	•	Split is time-based (no shuffling)

⸻

5. Model Training

Uses LightGBM Regressor:
	•	500 estimators
	•	Max depth = 6
	•	Learning rate = 0.05

⸻

6. Predictions & Evaluation

Metrics computed:
	•	📉 Mean Absolute Error (MAE)
	•	🎯 Directional Accuracy
	•	Measures how often the model predicts the correct price direction

⸻

7. Trading Strategy
	•	Signal = predicted price change
	•	Clipped between -1 and 1 (position sizing)
	•	Positive → long
	•	Negative → short

⸻

8. PnL Simulation

PnL formula:

PnL = signal × actual_price_move - transaction_cost

	•	Transaction cost: 0.05 per trade
	•	Cumulative PnL is tracked over time

⸻

9. Daily Performance
	•	Resamples PnL to daily frequency
	•	Computes:
	•	Daily profit/loss
	•	Yearly breakdown

⸻

10. Visualization

For each year:
	•	📊 Bar chart → daily PnL (green/red)
	•	📈 Line chart → cumulative PnL

⸻

11. Output Metrics

The script prints:
	•	Accuracy (direction prediction)
	•	Total PnL (€)
	•	Hit rate (profitable trades)
	•	Number of trades

⸻

▶️ How to Run

python your_script_name.py

Make sure Netherlands.csv is in the same directory.

⸻

📈 Example Output

Accuracy: 0.62
Total PnL: 1345.27
Hit rate: 0.58
Number of trades: 8760


⸻

🧠 Strategy Insights
	•	The model predicts magnitude + direction, not just classification
	•	Position sizing is proportional to confidence
	•	Transaction costs are included for realism
	•	Works as a baseline quantitative trading strategy

⸻

⚠️ Limitations
	•	Uses only price-based features (no external data)
	•	No hyperparameter tuning
	•	No risk management (e.g., stop-loss)
	•	Assumes perfect execution

⸻

🔮 Future Improvements
	•	Add weather, demand, and supply data
	•	Hyperparameter optimization (Optuna/Grid Search)
	•	Risk-adjusted metrics (Sharpe ratio, drawdown)
	•	More advanced models (LSTM, Transformers)
	•	Backtesting with realistic constraints

⸻

📄 License

MIT License

⸻

🙌 Acknowledgements
	•	LightGBM for efficient gradient boosting
	•	Scikit-learn for evaluation tools
	•	Pandas for time-series handling

⸻

💡 Notes
	•	Ensure no missing timestamps in your data
	•	Larger datasets improve model stability
	•	Tune transaction costs based on your market

⸻

Happy forecasting and trading! ⚡📈
