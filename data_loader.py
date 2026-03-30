from dataclasses import dataclass

import pandas as pd

from models import Market, PipelineConfig


@dataclass
class DataBundle:
    markets: list[Market]
    market_ids: list[str]           # canonical ordering — index i in any N×N matrix = market_ids[i]
    market_id_to_idx: dict[str, int]
    markets_df: pd.DataFrame
    prices_df: pd.DataFrame
    trades_df: pd.DataFrame

    def subset(self, indices: list[int]) -> "DataBundle":
        """Return a new DataBundle containing only the markets at the given indices."""
        sub_markets = [self.markets[i] for i in indices]
        sub_ids = [self.market_ids[i] for i in indices]
        sub_id_set = set(sub_ids)

        sub_markets_df = self.markets_df.iloc[indices].reset_index(drop=True)
        sub_prices_df = self.prices_df[
            self.prices_df["market_id"].isin(sub_id_set)
        ].reset_index(drop=True)
        sub_trades_df = self.trades_df[
            self.trades_df["market_id"].isin(sub_id_set)
        ].reset_index(drop=True)

        return DataBundle(
            markets=sub_markets,
            market_ids=sub_ids,
            market_id_to_idx={mid: i for i, mid in enumerate(sub_ids)},
            markets_df=sub_markets_df,
            prices_df=sub_prices_df,
            trades_df=sub_trades_df,
        )


class DataLoader:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def load(self) -> DataBundle:
        d = self.config.data_dir
        markets_df = pd.read_csv(f"{d}/markets.csv")
        prices_df = pd.read_csv(f"{d}/prices_daily.csv")
        trades_df = pd.read_csv(f"{d}/trades_daily.csv")

        markets = []
        for _, row in markets_df.iterrows():
            try:
                market = Market(
                    id=str(row["id"]),
                    condition_id=str(row.get("conditionId") or ""),
                    slug=str(row.get("slug") or ""),
                    question=str(row.get("question") or ""),
                    category=row.get("category") if pd.notna(row.get("category")) else None,
                    start_date=row.get("startDate"),
                    end_date=row.get("endDate"),
                    closed_time=row.get("closedTime"),
                    volume=float(row.get("volumeNum") or 0),
                    liquidity=float(row.get("liquidityNum") or 0),
                    tags=row.get("tags"),
                    outcomes=row.get("outcomes"),
                    outcome_prices=row.get("outcomePrices"),
                    clob_token_ids=row.get("clobTokenIds"),
                )
                markets.append(market)
            except Exception as e:
                print(f"  Warning: skipping malformed market row — {e}")
                continue

        market_ids = [m.id for m in markets]

        print(f"Loaded {len(markets)} valid markets")
        print(f"Price rows: {len(prices_df)}, Trade rows: {len(trades_df)}")

        return DataBundle(
            markets=markets,
            market_ids=market_ids,
            market_id_to_idx={mid: i for i, mid in enumerate(market_ids)},
            markets_df=markets_df,
            prices_df=prices_df,
            trades_df=trades_df,
        )
