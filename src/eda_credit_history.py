
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable

CREDIT_NUMERIC_COLS = ["dti", "dti_joint", "delinq_2yrs", "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
"open_acc", "total_acc", "pub_rec", "acc_now_delinq", "revol_bal", "revol_util", "total_rev_hi_lim",
"tot_coll_amt", "tot_cur_bal", "total_bal_il", "open_acc_6m", "open_il_6m", "open_il_12m", "open_il_24m",
"mths_since_rcnt_il", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
"inq_last_6mths", "inq_last_12m", "inq_fi", "collections_12_mths_ex_med"]

"""Goal:
Explore how credit history, balances, utilization, and inquiries 
relate to loan_status"""

class CreditHistoryEDA:
    def __init__(self, df: pd.DataFrame, target_col: str = "loan_status"):
        """Store the full DataFrame and the name of the target column."""
        self.df = df
        self.target_col = target_col

    def credit_structure_summary(self) -> pd.DataFrame:
        """
        One row per CREDIT_NUMERIC_COLS column with:
        - column
        - dtype
        - n_missing
        - missing_pct
        - mean (if numeric)
        - std (if numeric)
        """
        df_Numeric = self.df[CREDIT_NUMERIC_COLS].copy(deep=True)
        row = []
        for col in CREDIT_NUMERIC_COLS:
            series = df_Numeric[col]
            
            n_missing = series.isna().sum()
            missing_pct = (n_missing / len(series)) * 100
            #if pd.api.types.is_numeric_dtype(series):  # as meansioned above in docs "mean (if numeric)""
            mean_val = series.mean()
            std_val = series.std()
            #else:
            #    mean_val = None
            #    std_val = None
            row.append({"column": col, "dtype": str(series.dtypes), "n_missing": n_missing, "missing_pct": missing_pct, "mean": mean_val, "std": std_val})        
        return pd.DataFrame(row, columns=["column", "dtype", "n_missing", "missing_pct", "mean", "std" ]) 

    def default_rate_by_bucket(self, col: str, bins: int = 4): 
        """
        For a numeric credit column (e.g., dti, revol_util),
        create `bins` buckets and compute default rate per bucket.

        Return a DataFrame with columns:
        - bucket (interval)
        - n_loans
        - default_rate
        """
        #df_drop = self.df[[col]].dropna(subset=[col]).copy(deep=True)
        #print("df_drop OK")
        Buckets = pd.qcut(self.df[col], q=bins) # create interval bucket i.e 4
        #print("self.df[bucket] is OK")
        result = self.df.groupby(Buckets)[self.target_col].agg(n_loans="count", default_rate="mean").reset_index()
       # print("result is OK")
        return result
    #print("function is OK")

    def correlation_with_default(self) -> pd.Series:
        """
        Compute correlation of each numeric credit column with the target
        (assuming loan_status is encoded as 0/1).
        Return a Series indexed by column name.
        """
        correlation = {}
        for col in CREDIT_NUMERIC_COLS:
            #if pd.api.types.is_numeric_dtype(self.df[col]):
            correlation[col] = self.df[col].corr(self.df[self.target_col])
            #else:
            #    correlation[col] = None
        return pd.Series(correlation, name="correlation_with_default")
###################### part 2
def credit_history_report(eda:CreditHistoryEDA):
    steps: Dict[str, Callable[[], Any]] = {"structure_summary": eda.credit_structure_summary, "dti_buckets": lambda: eda.default_rate_by_bucket("dti", bins=5),
    "revol_util_buckets": lambda: eda.default_rate_by_bucket("revol_util", bins=5), "correlation_with_default": eda.correlation_with_default}
    report: Dict[str, Any] = {}
    for name, func in steps.items():
        report[name] = func()
    return report    




# # 2. Functional credit-history report
# # Add a functional report generator that coordinates several EDA steps:
# # """def credit_history_report(eda: CreditHistoryEDA) -> Dict[str, Any]:"""
# # Build a dict of step_name -> callable and run them to produce
# # a combined report.
# # Example steps:
# #   - "structure_summary": eda.credit_structure_summary
# #   - "dti_buckets": lambda: eda.default_rate_by_bucket("dti", bins=5)
# #   - "revol_util_buckets": lambda: eda.default_rate_by_bucket("revol_util", bins=5)
# #   - "correlation_with_default": eda.correlation_with_default
# # Iterate over this dict, call each function, and return
# # a result dict: step_name -> output.
# # Example idea:
# # """ def credit_history_report(eda: CreditHistoryEDA) -> Dict[str, Any]:
# # steps: Dict[str, Callable[[], Any]] = {
# # "structure_summary": eda.credit_structure_summary,
# # "dti_buckets": lambda: eda.default_rate_by_bucket("dti", bins=5),
# # "revol_util_buckets": lambda: eda.default_rate_by_bucket("revol_util", bins=5),
# # "correlation_with_default": eda.correlation_with_default,
# # }
# # report: Dict[str, Any] = {}
# # for name, func in steps.items():
# #     report[name] = func()
# # return report"""
# This should clearly show higher-order functions (functions stored and called later).