
import os
import pandas as pd

def preprocess_data():
    print("🚀 Memulai fungsi preprocess_data")

    # Buat folder
    output_dir = "bank_preprocessing"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Folder dibuat: {os.path.abspath(output_dir)}")

    # Simpan file dummy
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    df.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    print("📁 File dummy X_train.csv disimpan.")

    # List isi folder
    print("📂 Isi folder:")
    for file in os.listdir(output_dir):
        print(" -", file)

    print("✅ Selesai.")

if __name__ == "__main__":
    preprocess_data()
