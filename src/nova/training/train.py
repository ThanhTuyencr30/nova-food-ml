import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.nova.config import (
    MODEL_DIR, PROCESSED_DATA_PATH, RANDOM_STATE, REPORT_DIR,
    TARGET_COLUMN, TEST_SIZE, TEXT_COLUMN, VIETNAM_DATA_PATH,
)
from src.nova.features.text_features import (
    available_numeric_columns, build_preprocessor, make_features,
)

MODEL_FILENAME = "nova_pipeline.joblib"  # predict.py của TV4 đọc đúng tên này


def get_candidates(fast: bool = False):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
                "linear_svc": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000),
    }
    if not fast:
        models["random_forest"] = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        )
    return models


def make_pipeline(estimator, numeric_columns) -> Pipeline:
    return Pipeline([
        ("features", build_preprocessor(numeric_columns)),
        ("classifier", estimator),
    ])


def split_data(df: pd.DataFrame):
    """Chia train/test (TV3 có thể import hàm này để dùng đúng cùng một cách chia)."""
    return train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[TARGET_COLUMN]
    )


def score_on(model, df: pd.DataFrame, numeric_columns) -> dict:
    X, y = make_features(df, numeric_columns), df[TARGET_COLUMN]
    pred = model.predict(X)
    return {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, pred),
    }


def train_all(df: pd.DataFrame, vietnam_df: pd.DataFrame = None, cv: int = 3,
              fast: bool = False, tune: bool = True):
    """Giữ nguyên kiểu trả về cũ:
    (best_name, best_score, results_df, X_test, y_test, best_model)
    X_test giờ là DataFrame (cột text + cột dinh dưỡng) để model.predict(X_test) chạy được.
    """
    train_df, test_df = split_data(df)
    numeric_cols = available_numeric_columns(train_df)
    print(f"train={len(train_df)} test={len(test_df)} "
          f"vietnam={0 if vietnam_df is None else len(vietnam_df)} | cột dinh dưỡng dùng: {numeric_cols}")

    y_train = train_df[TARGET_COLUMN]
    feature_sets = {"text_only": []}
    if numeric_cols:
        feature_sets["text+nutrition"] = numeric_cols

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE) if cv > 1 else None
    rows, fitted = [], {}

    for feat_name, ncols in feature_sets.items():
        X_train = make_features(train_df, ncols)
        for name, estimator in get_candidates(fast).items():
            t0 = time.time()
            pipe = make_pipeline(estimator, ncols)
            cv_f1 = (cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro").mean()
                     if skf else float("nan"))
            pipe.fit(X_train, y_train)
            test_res = score_on(pipe, test_df, ncols)
            row = {"model": name, "features": feat_name, "cv_macro_f1": cv_f1,
                   "macro_f1": test_res["macro_f1"], "accuracy": test_res["accuracy"]}
            if vietnam_df is not None and len(vietnam_df):
                row["vietnam_macro_f1"] = score_on(pipe, vietnam_df, ncols)["macro_f1"]
            row["train_time_s"] = round(time.time() - t0, 1)
            rows.append(row)
            fitted[(name, feat_name)] = (pipe, ncols)
            print(row)

    results = pd.DataFrame(rows)
    # Chọn theo CV (không nhìn tập test); nếu không có CV thì chọn theo test.
    sort_col = "cv_macro_f1" if skf else "macro_f1"
    best = results.sort_values(sort_col, ascending=False).iloc[0]
    best_name, best_feat = best["model"], best["features"]
    best_pipe, best_cols = fitted[(best_name, best_feat)]
    print(f"\nMô hình tốt nhất (theo {sort_col}): {best_name} / {best_feat}")

    tuned = {}
    if tune and skf:
        grid = {"features__tfidf__ngram_range": [(1, 1), (1, 2)], "features__tfidf__min_df": [2, 5]}
        if best_name in ("logistic_regression", "linear_svc"):
            grid["classifier__C"] = [0.1, 0.5, 1, 5]
        elif best_name == "random_forest":
            grid["classifier__max_depth"] = [None, 30]
        gs = GridSearchCV(make_pipeline(get_candidates(fast)[best_name], best_cols), grid,
                          scoring="f1_macro", cv=skf, n_jobs=1, verbose=1, refit=True)
        gs.fit(make_features(train_df, best_cols), y_train)
        best_pipe = gs.best_estimator_
        tuned = {k: str(v) for k, v in gs.best_params_.items()}
        print("Tham số tốt nhất:", tuned, "| CV macro-F1:", round(gs.best_score_, 4))
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "rank_test_score"]]
         .sort_values("rank_test_score").to_csv(REPORT_DIR / "tuning_results.csv", index=False))

    final = score_on(best_pipe, test_df, best_cols)
    vn_final = (score_on(best_pipe, vietnam_df, best_cols)
                if vietnam_df is not None and len(vietnam_df) else None)

    # ---- lưu model, bảng so sánh, dự đoán (cho TV3, TV4) ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_DIR / MODEL_FILENAME)
    results.sort_values("macro_f1", ascending=False).round(4) \
        .to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    meta = {"model": best_name, "features": best_feat, "numeric_columns": best_cols,
            "tuned_params": tuned, "test": final, "vietnam": vn_final,
            "random_state": RANDOM_STATE, "test_size": TEST_SIZE}
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    X_test, y_test = make_features(test_df, best_cols), test_df[TARGET_COLUMN]
    out = test_df.drop(columns=[c for c in test_df.columns if c not in ("code", "product_name", TEXT_COLUMN)],
                       errors="ignore").copy()
    out["y_true"], out["y_pred"] = y_test.values, best_pipe.predict(X_test)
    out.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    if vietnam_df is not None and len(vietnam_df):
        vout = vietnam_df.drop(columns=[c for c in vietnam_df.columns
                                        if c not in ("code", "product_name", TEXT_COLUMN)], errors="ignore").copy()
        vout["y_true"] = vietnam_df[TARGET_COLUMN].values
        vout["y_pred"] = best_pipe.predict(make_features(vietnam_df, best_cols))
        vout.to_csv(REPORT_DIR / "vietnam_predictions.csv", index=False)

    return best_name, final["macro_f1"], results, X_test, y_test, best_pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROCESSED_DATA_PATH))
    ap.add_argument("--vn-data", default=str(VIETNAM_DATA_PATH))
    ap.add_argument("--cv", type=int, default=3, help="số fold CV (0 = bỏ CV)")
    ap.add_argument("--sample", type=int, default=0, help="lấy mẫu N dòng để chạy thử")
    ap.add_argument("--fast", action="store_true", help="bỏ RandomForest")
    ap.add_argument("--no-tune", action="store_true", help="bỏ bước tinh chỉnh siêu tham số")
    args = ap.parse_args()

    from pathlib import Path
    data_path, vn_path = Path(args.data), Path(args.vn_data)
    if not data_path.exists():
        raise FileNotFoundError(f"Chưa có {data_path}. Chạy module prepare trước "
                                f"(hoặc dùng dữ liệu giả: python -m src.nova.data.make_fake_data).")
    df = pd.read_csv(data_path)
    if args.sample and len(df) > args.sample:
        df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
    vietnam_df = pd.read_csv(vn_path) if vn_path.exists() else None
    if vietnam_df is not None and len(vietnam_df) == 0:
        vietnam_df = None
    print("Phân bố nhãn:\n", df[TARGET_COLUMN].value_counts().sort_index().to_string())

    best_name, score, results, X_test, y_test, model = train_all(
        df, vietnam_df, cv=args.cv, fast=args.fast, tune=not args.no_tune
    )
    print("\n=== BẢNG SO SÁNH ===")
    print(results.sort_values("macro_f1", ascending=False).round(4).to_string(index=False))
    print(f"\nBest model: {best_name} | Macro-F1 (test): {score:.4f}")
    print(classification_report(y_test, model.predict(X_test), zero_division=0))
    print(f"Đã lưu: {MODEL_DIR / MODEL_FILENAME}, model_meta.json và các file trong {REPORT_DIR}")


if __name__ == "__main__":
    main()