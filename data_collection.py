# data collection script, worked on with ChatGPT.

import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests


GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

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


def unix_to_date(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).date().isoformat()


def get_json(url, params=None, sleep_s=0.15, timeout=30):
    r = SESSION.get(url, params=params, timeout=timeout)
    time.sleep(sleep_s)  # simple rate limit
    r.raise_for_status()
    return r.json()


def fetch_markets_from_gamma(
    start_date_min="2023-01-01T00:00:00Z",
    closed=True,                 # closed markets for "ground truth"
    volume_num_min=10_000,        # your doc’s liquidity filter
    limit_total=300,
):
    markets = []
    offset = 0
    page_size = 100

    while len(markets) < limit_total:
        batch = get_json(
            f"{GAMMA}/markets",
            params={
                "closed": str(closed).lower(),
                "archived": "false",
                "start_date_min": start_date_min,
                "volume_num_min": volume_num_min,
                "order": "startDate",
                "ascending": "false",
                "limit": page_size,
                "offset": offset,
            },
        )
        if not batch:
            break
        markets.extend(batch)
        offset += page_size

    return markets[:limit_total]


def fetch_price_history(token_id: str, fidelity=1440, interval="max"):
    data = get_json(
        f"{CLOB}/prices-history",
        params={"market": token_id, "interval": interval, "fidelity": fidelity},
    )
    # CLOB returns {"history": [...]}
    return data.get("history", []) if isinstance(data, dict) else []


def extract_to_csv(
    start_date_min="2023-01-01T00:00:00Z",
    limit_total=300,
    volume_num_min=10_000,
    output_dir=".",
    price_fidelity=1440,  # daily snapshots
):
    markets = fetch_markets_from_gamma(
        start_date_min=start_date_min,
        closed=True,
        volume_num_min=volume_num_min,
        limit_total=limit_total,
    )

    # --- markets.csv ---
    market_rows = []
    for m in markets:
        market_rows.append(
            {
                "id": m.get("id"),
                "conditionId": m.get("conditionId"),
                "slug": m.get("slug"),
                "question": m.get("question"),
                "category": m.get("category"),
                "startDate": m.get("startDate"),
                "endDate": m.get("endDate"),
                "closedTime": m.get("closedTime"),
                "volumeNum": m.get("volumeNum") if m.get("volumeNum") is not None else m.get("volume"),
                "liquidityNum": m.get("liquidityNum") if m.get("liquidityNum") is not None else m.get("liquidity"),
                "tags": json.dumps(safe_loads(m.get("tags"), []), ensure_ascii=False),
                "outcomes": json.dumps(safe_loads(m.get("outcomes"), []), ensure_ascii=False),
                "outcomePrices": json.dumps(safe_loads(m.get("outcomePrices"), []), ensure_ascii=False),
                "clobTokenIds": json.dumps(safe_loads(m.get("clobTokenIds"), []), ensure_ascii=False),
            }
        )

    markets_df = pd.DataFrame(market_rows)
    markets_path = f"{output_dir.rstrip('/')}/markets.csv"
    markets_df.to_csv(markets_path, index=False)

    # --- prices_daily.csv ---
    price_rows = []
    for idx, m in enumerate(markets):
        token_ids = safe_loads(m.get("clobTokenIds"), [])
        outcomes = safe_loads(m.get("outcomes"), [])

        # Map token -> outcome name if lengths match (often YES/NO)
        token_to_outcome = {}
        if len(token_ids) == len(outcomes):
            token_to_outcome = {token_ids[i]: outcomes[i] for i in range(len(token_ids))}

        for token_id in token_ids:
            hist = fetch_price_history(token_id, fidelity=price_fidelity, interval="max")
            for pt in hist:
                price_rows.append(
                    {
                        "market_id": m.get("id"),
                        "conditionId": m.get("conditionId"),
                        "slug": m.get("slug"),
                        "token_id": token_id,
                        "outcome": token_to_outcome.get(token_id, ""),
                        "date_utc": unix_to_date(pt["t"]) if "t" in pt else None,
                        "timestamp": pt.get("t"),
                        "price": pt.get("p"),
                    }
                )

        if (idx + 1) % 25 == 0:
            print(f"Processed {idx+1}/{len(markets)} markets...")

    prices_df = pd.DataFrame(price_rows)
    prices_path = f"{output_dir.rstrip('/')}/prices_daily.csv"
    prices_df.to_csv(prices_path, index=False)

    print("Wrote:")
    print(" -", markets_path)
    print(" -", prices_path)


if __name__ == "__main__":
    extract_to_csv(
        start_date_min="2023-01-01T00:00:00Z",
        limit_total=300,
        volume_num_min=10_000,
        output_dir=".",
        price_fidelity=1440,
    )