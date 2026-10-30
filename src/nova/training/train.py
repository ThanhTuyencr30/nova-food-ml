import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

from src.nova.config import PROCESSED_DATA_PATH, MODEL_DIR, TEXT_COLUMN, TARGET_COLUMN, RANDOM_STATE, TEST_SIZE

def get_candidates():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "linear_svc": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

def train_all(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN], df[TARGET_COLUMN],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
    )
    results = []
    best_name, best_model, best_score = None, None, -1

    for name, estimator in get_candidates().items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                min_df=2, max_features=100_000, sublinear_tf=True
            )),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        score = f1_score(y_test, predictions, average="macro")
        results.append({"model": name, "macro_f1": score})
        if score > best_score:
            best_name, best_model, best_score = name, pipeline, score

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "nova_pipeline.joblib")
    return best_name, best_score, pd.DataFrame(results), X_test, y_test, best_model

def main():
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError("Chưa có dữ liệu đã xử lý. Chạy module prepare trước.")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    best_name, score, results, X_test, y_test, model = train_all(df)
    print(results.sort_values("macro_f1", ascending=False).to_string(index=False))
    print(f"Best model: {best_name} | Macro-F1: {score:.4f}")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))

if __name__ == "__main__":
    main()
