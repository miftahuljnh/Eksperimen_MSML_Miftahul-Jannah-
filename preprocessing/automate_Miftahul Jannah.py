from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
import pandas as pd
import os

def auto_preprocess_diabetes(
    data,
    target_col,
    test_size=0.3,
    output_folder="namadataset_preprocessing"
):
    # Pastikan folder output ada
    os.makedirs(output_folder, exist_ok=True)

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    X_train_scaled = pd.DataFrame(pipeline.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(pipeline.transform(X_test), columns=X.columns)

    # Simpan file
    dump(pipeline, os.path.join(output_folder, "scaler_diabetes.joblib"))
    X_train_scaled.to_csv(os.path.join(output_folder, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(output_folder, "X_test_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(output_folder, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_folder, "y_test.csv"), index=False)

    print(f"Preprocessing selesai! File disimpan di: {os.path.abspath(output_folder)}")
    return X_train_scaled, X_test_scaled, y_train, y_test

# Pemanggilan
current_dir = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(current_dir, "..", "namadataset_raw", "diabetes.csv")
out_dir = os.path.join(current_dir, "namadataset_preprocessing")

if os.path.exists(input_csv):
    df = pd.read_csv(input_csv)
    auto_preprocess_diabetes(data=df, target_col="Outcome", output_folder=out_dir)
else:
    print(f"Error: File tidak ditemukan di {input_csv}")