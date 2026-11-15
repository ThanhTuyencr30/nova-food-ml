from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "nova_pipeline.joblib"

NOVA_DESCRIPTIONS = {
    1: "Chưa chế biến hoặc ít chế biến",
    2: "Nguyên liệu thực phẩm chế biến",
    3: "Thực phẩm chế biến",
    4: "Thực phẩm siêu chế biến",
}

_model = None


def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Chưa có mô hình. Hãy huấn luyện trước.")
        _model = joblib.load(MODEL_PATH)
    return _model


def _build_input(model, ingredients: str, nutrients: dict = None) -> pd.DataFrame:
    """Tạo DataFrame đúng các cột mà model đã học (text, và cột dinh dưỡng nếu có)."""
    nutrients = nutrients or {}
    row = {}
    for col in model.feature_names_in_:
        row[col] = str(ingredients or "") if col == "ingredients_text" else nutrients.get(col, np.nan)
    return pd.DataFrame([row])


def predict_ingredients(ingredients: str, nutrients: dict = None):
    """Giữ nguyên giao diện cũ: trả về (nhãn NOVA, mô tả).
    nutrients (tùy chọn): {"sugars_100g": 12, "fat_100g": 3, ...}
    """
    model = load_model()
    label = int(model.predict(_build_input(model, ingredients, nutrients))[0])
    return label, NOVA_DESCRIPTIONS.get(label, "Không xác định")


def predict_proba_ingredients(ingredients: str, nutrients: dict = None) -> dict:
    """Trả về {1: p1, 2: p2, 3: p3, 4: p4} (xác suất, hoặc softmax điểm với LinearSVC)."""
    model = load_model()
    X = _build_input(model, ingredients, nutrients)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
    else:
        d = model.decision_function(X)[0]
        e = np.exp(d - d.max())
        p = e / e.sum()
    return {int(c): float(v) for c, v in zip(model.classes_, p)}