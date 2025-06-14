import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(path):
    try:
        print("🚀 Memulai fungsi preprocess_data")

        # Load data
        df = pd.read_csv(path)
        print("📥 Data dimuat, shape:", df.shape)

        # Drop duplikat dan kolom tidak relevan
        df.drop_duplicates(inplace=True)
        df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
        print("🧹 Kolom dibersihkan, shape:", df.shape)

        # One-hot encoding
        df = pd.get_dummies(df, columns=['Gender', 'Geography', 'Card Type'], drop_first=True)
        print("🔤 One-hot encoding selesai, kolom:", df.columns.tolist())

        # Scaling
        scaler = StandardScaler()
        scaled_cols = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary', 'Point Earned']
        df[scaled_cols] = scaler.fit_transform(df[scaled_cols])
        print("📏 Scaling selesai")

        # Remove outliers
        def remove_outliers_iqr(df, cols):
            for col in cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            return df

        df = remove_outliers_iqr(df, scaled_cols)
        print("⚠️ Outlier ditangani, shape:", df.shape)

        # Pisahkan fitur dan target
        X = df.drop(columns=['Exited'])
        y = df['Exited']

        # SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print("⚖️ SMOTE selesai, distribusi:")
        print(y_res.value_counts())

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

        # Simpan hasil ke folder
        output_dir = "bank_preprocessing"
        os.makedirs(output_dir, exist_ok=True)
        print("✅ Folder dibuat:", os.path.abspath(output_dir))

        pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        pd.concat([X_res, y_res], axis=1).to_csv("bankdataset_preprocessed.csv", index=False)
        print("📁 Semua file berhasil disimpan di:", output_dir)

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print("❌ ERROR:", str(e))

if __name__ == "__main__":
    preprocess_data("bankdataset_raw.csv")