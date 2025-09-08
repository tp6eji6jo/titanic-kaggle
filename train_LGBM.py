import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")
OUTPUT_CSV = "LightGBM_submission.csv"

TARGET = "Survived"

y = train[TARGET]
X = train.drop(columns=[TARGET])

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    #device="gpu",
    random_state=42
)

clf = Pipeline(steps=[("preprocess", preprocess),
                     ("model", model)])

clf.fit(X, y)

test_pred = clf.predict(test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred
})

submission.to_csv(OUTPUT_CSV, index=False)
print(f"已輸出: {OUTPUT_CSV}")