#  Loads all three CSVs and prints summary stats
# to verify the data collection worked correctly before building features

import pandas as pd
import json

m  = pd.read_csv('markets.csv')
p  = pd.read_csv('prices_daily.csv')
t  = pd.read_csv('trades_daily.csv')

print("Markets:")
print(f"Total markets: {len(m)}")
print(f"Categories: {m['category'].value_counts().to_dict()}")
print()

print("Prices:")
print(f"Total price rows: {len(p)}")
print(f"Markets with price data: {p['market_id'].nunique()}")
price_per_market = p.groupby('market_id').size()
print(f"Avg price points per market: {price_per_market.mean():.1f}")
print(f"Min price points: {price_per_market.min()}")
print(f"Max price points: {price_per_market.max()}")
print()

print("Trades:")
print(f"Total trade rows: {len(t)}")
print(f"Markets with trade data: {t['market_id'].nunique()}")
trade_per_market = t.groupby('market_id').size()
print(f"Avg trade days per market: {trade_per_market.mean():.1f}")
print()

print("Sample markets:")
print(m[['question', 'category', 'volumeNum']].head(10).to_string())