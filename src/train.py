import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler as skStandardScaler
from sklearn.model_selection import train_test_split as sktrain_test_split
from sklearn.ensemble import RandomForestClassifier as skRandomClassifier

scaler = skStandardScaler()
data = pd.read_csv("data/processed_data.csv")

numeric_cols = data.select_dtypes(include=['int64', 'float64'])
numeric_cols = numeric_cols.columns.difference(['income'])
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])


X = data.drop(columns=["workclass", "fnlwgt", "education",
                       "education-num", "marital-status", "occupation",
                       "relationship", "race", "capital-loss"])

y = data['income'].astype('int32')

X_train, X_test, y_train, y_test = sktrain_test_split(
      X, y, train_size=0.8, random_state=42)
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

sk_rf = skRandomClassifier(n_estimators=100, random_state=42)
sk_rf.fit(X_train_np, y_train_np)
joblib.dump(sk_rf, 'models/house_price_model.pkl')