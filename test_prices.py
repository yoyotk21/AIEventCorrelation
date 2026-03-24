# scans markets to find ones with actual price history,
# used to debug why the CLOB API was returning empty data for most markets

import requests
import pandas as pd
import json

m = pd.read_csv('markets.csv')

found = 0
checked = 0

for _, row in m.iterrows():
    tokens = json.loads(row['clobTokenIds'])
    if not tokens:
        continue

    token = tokens[0]
    r = requests.get('https://clob.polymarket.com/prices-history',
        params={'market': token, 'interval': 'max', 'fidelity': 1440})
    
    data = r.json()
    hist = data.get('history', []) if isinstance(data, dict) else []
    question = str(row['question'])[:60]

    if len(hist) > 0:
        print(f"FOUND DATA: {len(hist)} points")
        print(f"  Question: {question}")
        print(f"  Token: {token}")
        found += 1
        if found >= 3:
            break
    else:
        print(f"Empty: {question}")

    checked += 1
    if checked >= 30:
        print("Checked 30 markets, stopping.")
        break

print(f"\nSummary: found price data in {found} of {checked} markets checked.")