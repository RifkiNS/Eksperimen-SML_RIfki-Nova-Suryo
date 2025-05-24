import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset dari file CSV."""
    df = pd.read_csv(file_path)
    return df

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris dengan nilai kosong (missing)."""
    return df.dropna()

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus outlier menggunakan metode IQR."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Melakukan standardisasi pada fitur numerik."""
    numeric_cols = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def preprocess_data(file_path: str, target_column: str):
    df = load_data(file_path)
    df = clean_missing_values(df)
    df = remove_outliers(df)
    df = standardize_features(df)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

if __name__ == "__main__":
    file_path = "data.csv"
    target_column = "label"
    X, y = preprocess_data(file_path, target_column)
