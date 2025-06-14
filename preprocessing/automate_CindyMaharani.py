
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(path):
    # Load data
    df = pd.read_csv(path)

    # Hapus duplikat
    df.drop_duplicates(inplace=True)

    # Hapus kolom non-prediktif
    df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Gender', 'Geography', 'Card Type'], drop_first=True)

    # Scaling fitur numerik
    scaler = StandardScaler()
    scaled_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Point Earned']
    df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

    # Pisahkan fitur dan target
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Simpan hasil preprocessing ke file
    df.to_csv("bankdataset_preprocessed.csv", index=False)

    return X, y
