# # data collection script, worked on with ChatGPT.

import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

GAMMA = "https://gamma-api.polymarket.com"
CLOB  = "https://clob.polymarket.com"
DATA  = "https://data-api.polymarket.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "pm-corr-detect/0.1"})



def safe_loads(x, default):
    """Gamma often returns JSON-as-strings like '["Yes","No"]'."""
    if x is None:
        return default
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return default
    return default


def unix_to_date(ts) -> str:
    '''Convert unix timestamp to ISO date string (UTC).'''
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()


def get_json(url, params=None, sleep_s=0.2, timeout=30):
    '''makes a GET request and returns the response as python list/dict'''
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        time.sleep(sleep_s)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  Warning: skipping failed request to {url} — {e}")
        return None


# API fetchers

def fetch_markets_from_gamma(
    start_date_min="2023-01-01T00:00:00Z",
    volume_num_min=10_000,
    limit_total=500,
):
    '''fetches closed markets from Gamma API in batches of 100,
    sorts by volume to get long running markets,
    filters out markets with less than 10K in volume'''
    markets, offset = [], 0
    while len(markets) < limit_total:
        batch = get_json(
            f"{GAMMA}/markets",
            params={
                "closed":         "true",
                "archived":       "false",
                "start_date_min": start_date_min,
                "volume_num_min": volume_num_min,
                "order":          "volume",
                "ascending":      "false",
                "limit":          100,
                "offset":         offset,
            },
        )
        if not batch:
            break
        markets.extend(batch)
        offset += 100
    return markets[:limit_total]


def fetch_price_history(token_id: str, fidelity=1440, interval="max"):
    '''fetches the full daily price history for one market outcome token,
    returns a list of {t: timestamp, p: price}'''
    data = get_json(
        f"{CLOB}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": fidelity},
    )
    if data is None:
        return []
    return data.get("history", []) if isinstance(data, dict) else []


def fetch_trade_activity(condition_id: str):
    '''fetches all indv trades for a market condition from the DATA API'''
    trades, offset = [], 0
    while True:
        batch = get_json(
            f"{DATA}/trades",
            params={"market": condition_id, "limit": 100, "offset": offset},
        )
        if not batch:
            break
        trades.extend(batch)
        offset += 100
        if len(batch) < 100:
            break
    return trades



def get_duration(m):
    '''returns how many days a market was active, 
    used to filter out short lived market that don't have enough price history'''
    try:
        start = datetime.fromisoformat(m['startDate'].replace('Z', '+00:00'))
        end_str = m['closedTime'] or m['endDate']
        end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        return (end - start).days
    except:
        return 0


def aggregate_trades_to_daily(trades: list, market_id: str, condition_id: str) -> list:
    ''' collapes raw trade records into one row per day per market,
    each raw trade had timestamp and size, 
     sum all trade sizes within each day to get total daily volume'''
    if not trades:
        return []

    rows = []
    for t in trades:
        ts = t.get("timestamp") or t.get("createdAt") or t.get("ts")
        size = t.get("size") or t.get("usdcSize") or 0
        if ts is None:
            continue
        rows.append({"date_utc": unix_to_date(ts), "size": float(size)})

    if not rows:
        return []

    df = pd.DataFrame(rows)
    daily = df.groupby("date_utc")["size"].sum().reset_index()
    daily.columns = ["date_utc", "daily_volume"]
    daily["market_id"]    = market_id
    daily["condition_id"] = condition_id
    return daily.to_dict("records")


'''main function to run the data collection and save to CSV files.'''
def extract_to_csv(
    start_date_min="2023-01-01T00:00:00Z",
    limit_total=500,
    volume_num_min=10_000,
    output_dir=".",
    price_fidelity=1440,
):
    print(f"Fetching up to {limit_total} markets from Gamma API...")
    markets = fetch_markets_from_gamma(
        start_date_min=start_date_min,
        volume_num_min=volume_num_min,
        limit_total=limit_total,
    )
    print(f"Got {len(markets)} markets.")

    # Filter to markets that ran for at least 14 days
    markets = [m for m in markets if get_duration(m) >= 14]
    print(f"After duration filter (14+ days): {len(markets)} markets")

    # markets.csv 
    market_rows = []
    for m in markets:
        market_rows.append({
            "id":            m.get("id"),
            "conditionId":   m.get("conditionId"),
            "slug":          m.get("slug"),
            "question":      m.get("question"),
            "category":      m.get("category"),
            "startDate":     m.get("startDate"),
            "endDate":       m.get("endDate"),
            "closedTime":    m.get("closedTime"),
            "volumeNum":     m.get("volumeNum") or m.get("volume"),
            "liquidityNum":  m.get("liquidityNum") or m.get("liquidity"),
            "tags":          json.dumps(safe_loads(m.get("tags"), []),           ensure_ascii=False),
            "outcomes":      json.dumps(safe_loads(m.get("outcomes"), []),       ensure_ascii=False),
            "outcomePrices": json.dumps(safe_loads(m.get("outcomePrices"), []),  ensure_ascii=False),
            "clobTokenIds":  json.dumps(safe_loads(m.get("clobTokenIds"), []),   ensure_ascii=False),
        })

    pd.DataFrame(market_rows).to_csv(f"{output_dir}/markets.csv", index=False)
    print(f"Saved markets.csv ({len(market_rows)} rows)")

    # prices_daily.csv + 3. trades_daily.csv 
    price_rows, trade_rows = [], []

    for idx, m in enumerate(markets):
        mid          = m.get("id")
        condition_id = m.get("conditionId")
        token_ids    = safe_loads(m.get("clobTokenIds"), [])
        outcomes     = safe_loads(m.get("outcomes"), [])

        token_to_outcome = {}
        if len(token_ids) == len(outcomes):
            token_to_outcome = {token_ids[i]: outcomes[i] for i in range(len(token_ids))}

        # Price history (one row per token per day) 
        for token_id in token_ids:
            hist = fetch_price_history(token_id, fidelity=price_fidelity)
            for pt in hist:
                price_rows.append({
                    "market_id":   mid,
                    "conditionId": condition_id,
                    "token_id":    token_id,
                    "outcome":     token_to_outcome.get(token_id, ""),
                    "date_utc":    unix_to_date(pt["t"]) if "t" in pt else None,
                    "timestamp":   pt.get("t"),
                    "price":       pt.get("p"),
                })

        # Trade activity (aggregated to daily volume) 
        if condition_id:
            raw_trades = fetch_trade_activity(condition_id)
            daily_vol  = aggregate_trades_to_daily(raw_trades, mid, condition_id)
            trade_rows.extend(daily_vol)

        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx+1}/{len(markets)} markets...")

    pd.DataFrame(price_rows).to_csv(f"{output_dir}/prices_daily.csv", index=False)
    pd.DataFrame(trade_rows).to_csv(f"{output_dir}/trades_daily.csv", index=False)

    print(f"Saved prices_daily.csv ({len(price_rows)} rows)")
    print(f"Saved trades_daily.csv ({len(trade_rows)} rows)")
    print("Done!")


if __name__ == "__main__":
    extract_to_csv(
        start_date_min="2023-01-01T00:00:00Z",
        limit_total=500,
        volume_num_min=10_000,
        output_dir=".",
        price_fidelity=1440,
    )