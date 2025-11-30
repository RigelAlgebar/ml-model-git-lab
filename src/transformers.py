# In src/transformers.py, implement Subset2CategoricalPerformanceTransformer:
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class Subset2CategoricalPerformanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols: list[str], min_count: int = 100):
        self.cat_cols = cat_cols
        self.min_count = min_count
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        self.rare_maps_ = {}  # mapping per column for rare categories

    def fit(self, X, y=None):
        df = pd.DataFrame(X, columns[self.cat_cols]).copy()
        # 1) normalize strings
        df = normalize_cat_strings(df, self.cat_cols)
        # 2) fill missing with sentinel
        df = fill_cat_missing_with_sentinel(df, self.cat_cols)
        # 3) detect rare categories per column and store mapping
        for col in self.cat_cols:
            counts = df[col].value_counts()
            rare = counts[counts < self.min_count].index
            self.rare_maps_[col] = set(rare)
            df[col] = df[col].where(~df[col].isin(rare), "Other")
        # 4) fit encoder
        self.encoder.fit(df[self.cat_cols])
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.cat_cols).copy()
        # apply normalization, missing fill, and rare grouping using self.rare_maps_
        df = normalize_cat_strings(df, self.cat_cols)
        df = fill_cat_missing_with_sentinel(df, self.cat_cols)
        for col in self.cat_cols:
            rare = self.rare_maps_.get(col, set())
            df[col] = df[col].where(~df[col].isin(rare), "Other")
        encoded = self.encoder.transform(df[self.cat_cols])
        # text features
        df_text = df.copy()
        df_text["desc"] = df_text.get("desc", "")
        df_text["title"] = df_text.get("title", "")

        df_text = add_desc_length(df_text)
        df_text = add_title_word_count(df_text)

        text_features = df_text[["desc_len", "title_word_count"]].values.astype(float)

        # final output: encoded OHE + text numeric
        return np.hstack([encoded, text_features])  #  return encoded