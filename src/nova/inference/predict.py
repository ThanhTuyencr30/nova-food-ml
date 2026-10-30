from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "nova_pipeline.joblib"

NOVA_DESCRIPTIONS = {
    1: "Chưa chế biến hoặc ít chế biến",
    2: "Nguyên liệu thực phẩm chế biến",
    3: "Thực phẩm chế biến",
    4: "Thực phẩm siêu chế biến",
}

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Chưa có mô hình. Hãy huấn luyện trước.")
    return joblib.load(MODEL_PATH)

def predict_ingredients(ingredients: str):
    model = load_model()
    label = int(model.predict([ingredients])[0])
    return label, NOVA_DESCRIPTIONS.get(label, "Không xác định")
