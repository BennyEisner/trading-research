import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["daily_return"] = df_copy["Close"].pct_change()
    df_copy = df_copy.dropna(subset=["daily_return"])

    return df_copy.filter(items=["daily_return"])
