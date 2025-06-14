import os
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

    # Deteksi dan penanganan outlier dengan IQR
    def remove_outliers_iqr(df, cols):
        for col in cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            before = df.shape[0]
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            after = df.shape[0]
            print(f"{col}: Removed {before - after} outliers.")
        return df

    df = remove_outliers_iqr(df, scaled_cols)

    # Pisahkan fitur dan target
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    # Simpan hasil preprocessing ke file
    output_path = os.path.join(os.getcwd(), "bankdataset_preprocessed.csv")
    df.to_csv(output_path, index=False)

    print(f"Hasil preprocessing disimpan ke: {output_path}")
    
    return X, y
