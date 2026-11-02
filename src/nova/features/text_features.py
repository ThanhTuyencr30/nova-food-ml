import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.nova.config import TEXT_COLUMN

# Các cột dinh dưỡng (tùy chọn): chỉ cột nào có thật trong dữ liệu mới được dùng.
NUM_COLUMNS = [
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g",
    "sugars_100g", "salt_100g", "proteins_100g", "fiber_100g",
]


def build_tfidf():
    # Fit vectorizer trên tập train chỉ; không fit trên toàn bộ dữ liệu trước khi chia tập.
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )


def available_numeric_columns(df: pd.DataFrame) -> list:
    """Các cột dinh dưỡng có mặt và có ít nhất 1 giá trị số trong df."""
    cols = []
    for c in NUM_COLUMNS:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            cols.append(c)
    return cols


def make_features(df: pd.DataFrame, numeric_columns=None) -> pd.DataFrame:
    """DataFrame đầu vào cho pipeline: cột text + các cột số (ép về số)."""
    numeric_columns = numeric_columns or []
    X = pd.DataFrame({TEXT_COLUMN: df[TEXT_COLUMN].fillna("").astype(str).values})
    for c in numeric_columns:
        X[c] = pd.to_numeric(df[c], errors="coerce").values if c in df.columns else float("nan")
    return X


def build_preprocessor(numeric_columns=None) -> ColumnTransformer:
    """TF-IDF cho text (+ đặc trưng dinh dưỡng nếu có)."""
    transformers = [("tfidf", build_tfidf(), TEXT_COLUMN)]
    if numeric_columns:
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric, list(numeric_columns)))
    return ColumnTransformer(transformers)