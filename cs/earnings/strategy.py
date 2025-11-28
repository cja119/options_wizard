def earnings_calendar_spread(data, **kwargs):
        """Calendar spread around earnings: short near-term, long farther expiry."""
        entry_offset = kwargs.get("entry_offset", 1)
        exit_offset = kwargs.get("exit_offset", 0)
        exit_tolerance = kwargs.get("exit_tolerance", 3)
        call_delta = kwargs.get("call_delta", 0.35)
        put_delta = kwargs.get("put_delta", -0.35)
        ttm_short = kwargs.get("ttm_short", 14)
        ttm_long = kwargs.get("ttm_long", 30)
        position_size = kwargs.get("position", 1)

        cols = [
            "trade_exit_date",
            "position",
            "mid_price_entry",
            "mid_price_exit",
            "pnl",
            "theta_entry",
        ]
        if data.empty:
            return pd.DataFrame(index=data.index, columns=cols)

        base_index = data.index
        entries = []

        # --- entry candidates ---
        entry_window = sorted((0, entry_offset))
        exit_window = sorted((exit_offset, exit_offset + exit_tolerance))
        entry_mask = data["bdays_to_earnings"].between(*entry_window)
        exit_mask = data["bdays_since_earnings"].between(*exit_window)

        entry_df = data.loc[entry_mask].copy()
        exit_df = data.loc[exit_mask].copy()

        if entry_df.empty or exit_df.empty:
            return pd.DataFrame(index=base_index, columns=cols)

        def pick_leg(frame, ttm_filter, delta_filter):
            leg = frame.loc[ttm_filter(frame)].loc[delta_filter]
            if leg.empty:
                return leg
            # one contract per trade_date per call/put direction
            return (
                leg.reset_index()
                .sort_values("trade_date_idx")
                .groupby(["trade_date_idx", "call_put"])
                .apply(lambda g: g.loc[g["ttm"].idxmin()])
                .droplevel(-1)
            )

        short_calls = pick_leg(
            entry_df,
            lambda df: df["ttm"] <= ttm_short,
            lambda df: df["delta"] >= call_delta,
        )
        long_calls = pick_leg(
            entry_df,
            lambda df: df["ttm"] >= ttm_long,
            lambda df: df["delta"] >= call_delta,
        )
        short_puts = pick_leg(
            entry_df,
            lambda df: df["ttm"] <= ttm_short,
            lambda df: df["delta"] <= put_delta,
        )
        long_puts = pick_leg(
            entry_df,
            lambda df: df["ttm"] >= ttm_long,
            lambda df: df["delta"] <= put_delta,
        )
        def sanitize_leg(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            if not set(base_index.names).issubset(df.columns):
                df = df.reset_index()
            else:
                df = df.copy()
            df.index = pd.RangeIndex(len(df))
            return df

        short_calls = sanitize_leg(short_calls)
        long_calls = sanitize_leg(long_calls)
        short_puts = sanitize_leg(short_puts)
        long_puts = sanitize_leg(long_puts)

        join_keys = {"trade_date_idx"}

        def build_pairs(short_leg, long_leg, call_put):
            merged = short_leg.merge(
                long_leg,
                on=["trade_date_idx", "call_put"],
                suffixes=("_short", "_long"),
            )
            records = []
            for _, row in merged.iterrows():
                idx_short = tuple(
                    row[name] if name in join_keys else row[f"{name}_short"]
                    for name in base_index.names
                )
                idx_long = tuple(
                    row[name] if name in join_keys else row[f"{name}_long"]
                    for name in base_index.names
                )
                short_row = data.loc[idx_short].copy()
                long_row = data.loc[idx_long].copy()

                records.append(
                    {
                        "index": idx_short,
                        "trade_exit_date": None,
                        "position": -position_size,
                        "mid_price_entry": short_row["mid_price"],
                        "theta_entry": short_row.get("theta", np.nan),
                    }
                )
                records.append(
                    {
                        "index": idx_long,
                        "trade_exit_date": None,
                        "position": position_size,
                        "mid_price_entry": long_row["mid_price"],
                        "theta_entry": long_row.get("theta", np.nan),
                    }
                )
            return records

        entries.extend(build_pairs(short_calls, long_calls, "c"))
        entries.extend(build_pairs(short_puts, long_puts, "p"))

        if not entries:
            return pd.DataFrame(index=base_index, columns=cols)

        result = pd.DataFrame(entries)
        if result.empty:
            return pd.DataFrame(index=base_index, columns=cols)
        result = result.set_index("index").reindex(base_index)

        # --- exit matching ---
        exit_info = (
            exit_df.reset_index()
            .sort_values("bdays_since_earnings")
            .groupby(["strike_idx", "expiry_date_idx", "call_put_idx"], as_index=False)
            .first()[["strike_idx", "expiry_date_idx", "call_put_idx", "trade_date_idx", "mid_price"]]
            .rename(
                columns={
                    "trade_date_idx": "trade_exit_date",
                    "mid_price": "mid_price_exit",
                }
            )
        )

        result = result.reset_index()
        result = result.drop(columns=["trade_exit_date", "mid_price_exit"], errors="ignore")
        result = result.merge(
            exit_info,
            on=["strike_idx", "expiry_date_idx", "call_put_idx"],
            how="left",
        )
        result = result.set_index(base_index.names).reindex(base_index)

        result["pnl"] = result["position"] * (
            result["mid_price_exit"] - result["mid_price_entry"]
        )
        result = result.loc[:, ~result.columns.duplicated()]
        return result[cols].loc[result["mid_price_entry"].notna()]
   
