# NOVA Food Processing Classification

Dự đoán nhóm NOVA (1–4) từ danh sách thành phần thực phẩm, sử dụng dữ liệu Open Food Facts.

## 1. Công nghệ
- Python 3.10+
- pandas, numpy, scikit-learn
- XGBoost (tùy chọn)
- Streamlit
- OpenCV / pyzbar (quét mã vạch, tùy chọn)

## 2. Cấu trúc
```text
nova-food-classification/
├── app.py
├── requirements.txt
├── .gitignore
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── README.md
├── reports/
│   └── figures/
├── src/
│   └── nova/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── prepare.py
│       ├── features/
│       │   ├── __init__.py
│       │   └── text_features.py
│       ├── training/
│       │   ├── __init__.py
│       │   └── train.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── evaluate.py
│       └── inference/
│           ├── __init__.py
│           └── predict.py
├── scripts/
│   └── train.py
└── tests/
    └── test_prepare.py
```

## 3. Cài đặt
```bash
git clone <URL_REPOSITORY>
cd nova-food-classification

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 4. Chuẩn bị dữ liệu
Đặt dữ liệu đã tải từ Open Food Facts vào `data/raw/`. Tên cột đầu vào mặc định:
- `ingredients_text`: thành phần thực phẩm
- `nova_group`: nhãn NOVA (1, 2, 3 hoặc 4)
- `product_name`: tên sản phẩm (không bắt buộc)

Sau khi chuẩn hóa, dữ liệu được lưu tại `data/processed/off_nova.csv`.

```bash
python -m src.nova.data.prepare
```

## 5. Huấn luyện và đánh giá
```bash
python scripts/train.py
```

Mô hình và báo cáo được lưu vào `models/` và `reports/`.

## 6. Chạy ứng dụng
```bash
streamlit run app.py
```

## 7. Quy trình làm việc nhóm
- `main`: phiên bản ổn định.
- `dev`: nhánh tích hợp.
- `feature/data-eda`: dữ liệu và EDA.
- `feature/model-training`: huấn luyện mô hình.
- `feature/evaluation`: đánh giá và phân tích lỗi.
- `feature/streamlit-demo`: giao diện demo.

Mỗi thành viên tạo branch riêng, commit thường xuyên và tạo Pull Request vào `dev`. Chỉ merge `dev` vào `main` khi đã kiểm thử.

## 8. Phân công tham khảo
| Thành viên | Phụ trách | Thư mục chính |
|---|---|---|
| 1 | Dữ liệu, lọc sản phẩm Việt Nam, EDA | `src/nova/data/`, `notebooks/` |
| 2 | Vector hóa, huấn luyện và so sánh mô hình | `src/nova/features/`, `src/nova/training/` |
| 3 | Macro-F1, confusion matrix, error analysis | `src/nova/evaluation/`, `reports/` |
| 4 | Streamlit, tích hợp, báo cáo | `app.py`, `src/nova/inference/` |

## Lưu ý khoa học
Nhãn NOVA trên Open Food Facts có thể được tạo tự động từ quy tắc dựa trên thành phần. Vì vậy, mô hình có nguy cơ học lại quy tắc tạo nhãn thay vì phản ánh đánh giá độc lập. Báo cáo cần nêu rõ giới hạn này và ưu tiên Macro-F1 khi đánh giá.
