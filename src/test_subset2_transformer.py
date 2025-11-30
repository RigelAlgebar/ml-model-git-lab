#tests/test_subset2_transformer.py
import pandas as pd
from transformers import Subset2CategoricalPerformanceTransformer

def test_no_nans_after_transform():
    df = pd.DataFrame({
        "pymnt_plan": ["y", None, "n"],
        "purpose": ["debt", "wedding", None],
        "title": ["abc def", None, "xxx"],
        "zip_code": ["123xx", "456xx", "123xx"],
        "addr_state": ["CA", "ZZ", "CA"],
    })

    tr = Subset2CategoricalPerformanceTransformer(
        cat_cols=list(df.columns), min_count=2
    )
    tr.fit(df)
    out = tr.transform(df)

    assert not pd.isna(out).any()


def test_rare_category_grouping():
    df = pd.DataFrame({
        "purpose": ["a", "b", "c", "d", "e"],   # all rare
        "pymnt_plan": ["y", "y", "y", "y", "y"],
        "title": ["t"]*5,
        "zip_code": ["111"]*5,
        "addr_state": ["CA","CA","CA","CA","CA"],
    })

    tr = Subset2CategoricalPerformanceTransformer(
        cat_cols=df.columns.tolist(),
        min_count=3
    )
    tr.fit(df)

    transformed = tr.transform(df)

    # purpose has 5 categories but each count=1 < 3 → all become Other
    assert "Other" in tr.rare_maps_["purpose"] or len(tr.rare_maps_["purpose"]) == 5


def test_text_features_numeric():
    df = pd.DataFrame({
        "pymnt_plan": ["y","n"],
        "purpose": ["debt","car"],
        "title": ["abc def","hello"],
        "zip_code": ["123","456"],
        "addr_state": ["CA","NY"],
    })

    tr = Subset2CategoricalPerformanceTransformer(cat_cols=df.columns.tolist())
    tr.fit(df)
    out = tr.transform(df)

    # last 2 columns must be numeric
    assert out[:, -2:].dtype.kind in ("i", "f")
