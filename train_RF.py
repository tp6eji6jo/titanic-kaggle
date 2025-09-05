import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("C:/Users/User/Desktop/CGU/練習/dataset/train.csv")
test  = pd.read_csv("C:/Users/User/Desktop/CGU/練習/dataset/test.csv")
OUTPUT_CSV = "submission.csv"

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

model = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1
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