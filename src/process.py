import pandas as pd
import urllib.request
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # URL of the dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases"
    data_url = f"{data_url}/adult/adult.data"

    # Download the file
    file_path = 'adult.data'
    urllib.request.urlretrieve(data_url, file_path)

    # Now read the file with cuDF
    data = pd.read_csv(file_path, names=[
                            "age", "workclass", "fnlwgt", "education",
                            "education-num", "marital-status", "occupation",
                            "relationship", "race", "sex", "capital-gain",
                            "capital-loss", "hours-per-week",
                            "native-country", "income"
                        ],
                        na_values="?")
    data.to_csv("data/raw_data.csv")
    data = data.dropna()
    for column in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])

    data.to_csv("data/processed_data.csv")

if __name__ == '__main__':
    preprocess_data()