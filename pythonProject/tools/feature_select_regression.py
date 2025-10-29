# feature_select_regression.py
# Finds top-K features for a numeric label using f_regression after basic preprocessing.

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# --- EDIT THESE ---
CSV_PATH = "data/raw_data/aalborg_taxi_raw_data.csv"
LABEL = "taxi_pickups"     # numeric label for regression
DROP_ALWAYS = ["datetime", LABEL]   # drop target & leakage columns
K = 10
# -------------------

# Load
try:
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"])
except Exception:
    df = pd.read_csv(CSV_PATH)

y = df[LABEL]
X = df.drop(columns=DROP_ALWAYS, errors="ignore")

# Preprocess: impute numerics; impute+one-hot categoricals
num_sel = selector(dtype_include=np.number)
cat_sel = selector(dtype_exclude=np.number)

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_sel),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_sel),
    ],
    remainder="drop",
)

pipe = Pipeline([
    ("prep", preprocess),
    ("sel", SelectKBest(score_func=f_regression, k=K)),
])

X_new = pipe.fit_transform(X, y)

# Recover feature names after preprocessing
num_feats = X.columns[X.dtypes != "object"]
cat_feats = X.columns[X.dtypes == "object"]

oh = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"] \
     if len(cat_feats) else None
cat_out = oh.get_feature_names_out(cat_feats) if oh is not None else np.array([])

all_feats = np.r_[num_feats, cat_out]
scores = pipe.named_steps["sel"].scores_

top = (pd.DataFrame({"Feature": all_feats, "Score": scores})
         .sort_values("Score", ascending=False)
         .head(K))

print("\nTop features (regression, f_regression):")
print(top.to_string(index=False))
