#scans markets to find ones with actual price history,
# used to debug why the CLOB API was returning empty data for most markets

import pandas as pd
from datetime import datetime, timezone

m = pd.read_csv('markets.csv')

# Parse dates
m['startDate'] = pd.to_datetime(m['startDate'], utc=True, errors='coerce')
m['closedTime'] = pd.to_datetime(m['closedTime'], utc=True, errors='coerce')

# Compute duration in days
m['duration_days'] = (m['closedTime'] - m['startDate']).dt.days

print("Duration distribution:")
print(m['duration_days'].describe())
print()
print("Markets by duration bucket:")
buckets = [0, 1, 3, 7, 14, 30, 90, 365, 9999]
labels = ['<1d', '1-3d', '3-7d', '7-14d', '14-30d', '30-90d', '90-365d', '>1yr']
m['bucket'] = pd.cut(m['duration_days'], bins=buckets, labels=labels)
print(m['bucket'].value_counts().sort_index())
print()
print(f"Markets with 14+ days: {(m['duration_days'] >= 14).sum()}")
print(f"Markets with 30+ days: {(m['duration_days'] >= 30).sum()}")

# Show some examples of long-duration markets
print()
print("Sample long-duration markets (30+ days):")
long = m[m['duration_days'] >= 30][['question', 'duration_days', 'volumeNum', 'category']].head(10)
print(long.to_string())