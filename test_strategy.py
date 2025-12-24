import pandas as pd
import numpy as np
import utils

# Load data
df = utils.load_data('NFLX.csv')
df_processed = utils.process_data(df.copy())

# Run Strategy
bt_results = utils.backtest_ma_strategy(df_processed)
metrics = utils.calculate_strategy_metrics(bt_results)

print("Strategy Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")

print("\nSignal counts:")
print(bt_results['Signal'].value_counts())

print("\nFirst few rows of strategy results:")
print(bt_results[['Close', 'MA_20', 'MA_50', 'Signal', 'Cumulative_Market', 'Cumulative_Strategy']].head())

print("\nLast few rows of strategy results:")
print(bt_results[['Close', 'MA_20', 'MA_50', 'Signal', 'Cumulative_Market', 'Cumulative_Strategy']].tail())
