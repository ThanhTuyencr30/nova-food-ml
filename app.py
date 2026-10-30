import streamlit as st
from src.nova.inference.predict import predict_ingredients

st.set_page_config(page_title="NOVA Food Classifier", page_icon="🥗")
st.title("🥗 Dự đoán nhóm NOVA")
st.write("Nhập danh sách thành phần để mô hình dự đoán nhóm NOVA.")

ingredients = st.text_area(
    "Thành phần thực phẩm",
    placeholder="Ví dụ: sugar, wheat flour, vegetable oil, salt..."
)

if st.button("Dự đoán", type="primary"):
    if not ingredients.strip():
        st.warning("Vui lòng nhập thành phần thực phẩm.")
    else:
        try:
            label, description = predict_ingredients(ingredients)
            st.metric("Nhãn NOVA dự đoán", f"NOVA {label}")
            st.success(description)
            st.caption("Kết quả là dự đoán của mô hình, không thay thế đánh giá chuyên môn.")
        except FileNotFoundError as exc:
            st.error(str(exc))
