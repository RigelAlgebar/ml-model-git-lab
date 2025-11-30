import pandas as pd
def normalize_cat_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    # Lowercase + strip whitespace in the given categorical columns.
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(str).str.lower().str.strip()
    return df
    

def fill_cat_missing_with_sentinel(
    df: pd.DataFrame,
    cols: list[str],
    label: str = "Missing",) -> pd.DataFrame:
    # Replace NaNs in categorical columns with a sentinel category.
    df = df.copy()
    for c in cols:
        df[c] = df[c].fillna(label)
        df[c] = df[c].replace("", label)
    return df

def group_rare_categories(
    df: pd.DataFrame,
    col: str,
    min_count: int = 100,
) -> pd.DataFrame:
    # Replace rare categories in a single column by "Other".
    df = df.copy()
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[col] = df[col].where(~df[col].isin(rare), "Other")
    return df