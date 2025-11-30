import pandas as pd
def add_desc_length(df: pd.DataFrame) -> pd.DataFrame:
    # Create "desc_len" as character length of "desc" (or 0 if NaN).
    df = df.copy()
    df["desc_len"] = df["desc"].fillna("").astype(str).str.len()
    return df


def add_title_word_count(df: pd.DataFrame) -> pd.DataFrame:
    # Create "title_word_count" as word count of "title".
    df = df.copy()
    df["title_word_count"] = (
        df["title"].fillna("").astype(str).str.split().apply(len)
    )
    return df

###### Implement a functional composition helper:

def apply_cat_steps(
    df: pd.DataFrame,
    cols: list[str],
    steps: list,
) -> pd.DataFrame:
    # Apply a sequence of functions(df, cols) -> df to categorical columns.
    for step in steps:
        df = step(df, cols)
    return df
    