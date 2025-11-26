
#import the following 

import pandas as pd
import numpy as np

from typing import Dict, Any, List

# upload dataset as follows:

!pip install gdown --quiet

import gdown

# shared file URL (can be /file/d/.../view) 

url = "https://drive.google.com/uc?id=1EKPQm63-_9P9eZsFeQsGI9O2GuAFUxdk"

# Download it to a local file in Colab
output = "credit_risk.csv"
gdown.download(url, output, quiet=False)

# Now read it with pandas to see if everything is as expected 
df = pd.read_csv(output)
df.head(10)

#_______________________________________________________

NUMERIC_TERM_COLS = [
"loan_amnt", "funded_amnt", "funded_amnt_inv",
"int_rate", "installment",
"out_prncp", "out_prncp_inv",
"total_pymnt", "total_pymnt_inv",
"total_rec_prncp", "total_rec_int",
"total_rec_late_fee", "recoveries", "collection_recovery_fee",
"last_pymnt_amnt",
]

CATEGORICAL_TERM_COLS = [
"term", "grade", "sub_grade", "initial_list_status",
]

# create a class assess numerical columns
class LoanTermsEDA:
  def __init__(self, df: pd.DataFrame, target_col: str = "loan_status"):
    self.df = df
    self.target_col = target_col

#Store the full DataFrame and the name of the target column...

def numeric_terms_summary(self) -> pd.DataFrame:
  return self.df[NUMERIC_TERM_COLS].describe().T

#(one row per numeric term/cashflow column))
#(one row per numeric term/cashflow column))
# once dataframe is stored in ... eda = LoanTermsEDA(df, target_col="loan_status") 
#you can have summery (describe) via eda.numeric_terms_summary()



def distribution_stats_for_column(self, col: str) -> Dict[str, float]:
  #For a given numeric column return stats in a dict:
  numeric_col = self.df[col].dropna()
  numeric_col = numeric_col
  return {"mean": numeric_col.mean(), "std": numeric_col.std(), "min": numeric_col.min(), "max": numeric_col.max(), "q25": numeric_col.quantile(0.25), "q50": numeric_col.quantile(0.50), "q75": numeric_col.quantile(0.75)}

    # - std
    # - min
    # - max
    # - q25 (25% quantile)
    # - q50 (median)
    # - q75 (75% quantile)

    # eda.distribution_stats_for_column()... takes a postional argument inside (), as you are looking into stats distributions per column 



def default_rate_by_term(self) -> pd.Series:


    # Default rate (mean of target) per 'term' category.

  return self.df.groupby("term")[self.target_col].mean()

  #could not test this as it went on forever running 


def default_rate_by_grade(self) -> pd.Series:

  # Default rate per 'grade' (or per 'sub_grade', if you prefer).


  return self.df.groupby("grade")[self.target_col].mean()

#_____________________________________________________________________

# the function in a new cell outside the created class othwerwise it will cosidered as part of the class method. 


#   2. Functional per-column numeric summary
# Add a function that uses functional programming over numeric columns:

#***** did not work this way: 

# def summarize_numeric_terms(eda: LoanTermsEDA) -> Dict[str, Dict[str, float]]:
#   numeric_list = list(map(eda.distribution_stats_for_column, NUMERIC_TERM_COLS))
#   return dict(NUMERIC_TERM_COLS, numeric_list)


# works with provided code: 
def summarize_numeric_terms(eda: LoanTermsEDA) -> Dict[str, Dict[str, float]]:
  return {col: eda.distribution_stats_for_column(col) for col in NUMERIC_TERM_COLS} 


  #eda = LoanTermsEDA(df, target_col="loan_status")
  #per_column_stats = summarize_numeric_terms(eda)
   #per_column_stats

