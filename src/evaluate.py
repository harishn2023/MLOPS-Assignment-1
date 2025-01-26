import pandas as pd
import joblib
from sklearn.model_selection import train_test_split as sktrain_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error

def evaluate_model():
    data = pd.read_csv("data/processed_data.csv")

    X = data.drop(columns=["workclass", "fnlwgt", "education",
                        "education-num", "marital-status", "occupation",
                        "relationship", "race", "capital-loss"])

    y = data['income'].astype('int32')

    X_train, X_test, y_train, y_test = sktrain_test_split(
        X, y, train_size=0.8, random_state=42)
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()
    model = joblib.load('models/house_price_model.pkl')
    y_pred_sk = model.predict(X_test_np)

    # Calculate metrics using scikit-learn
    precision_sk = precision_score(y_test_np, y_pred_sk, average='macro')
    recall_sk = recall_score(y_test_np, y_pred_sk, average='macro')
    f1_sk = f1_score(y_test_np, y_pred_sk, average='macro')
    accuracy_sk = accuracy_score(y_test_np, y_pred_sk)
