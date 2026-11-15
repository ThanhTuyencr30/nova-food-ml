
import csv
import pandas as pd

from src.nova.config import RAW_DATA_PATH

CHUNKSIZE = 100000

columns = [
    "countries_tags",
    "countries_en",
    "ingredients_text",
    "nova_group",
]

vietnam_parts = []

print("Dang kiem tra du lieu nguon...")

reader = pd.read_csv(
    RAW_DATA_PATH,
    sep="\t",
    encoding="utf-8-sig",
    dtype=str,
    usecols=columns,
    chunksize=CHUNKSIZE,
    quoting=csv.QUOTE_NONE,
)

for chunk in reader:
    tags = chunk["countries_tags"].fillna("").str.lower()
    countries = chunk["countries_en"].fillna("").str.lower()

    # Nhận diện Việt Nam theo mã quốc gia
    is_vietnam = (
        tags.str.contains(
            r"(^|,)en:vietnam(,|$)",
            regex=True,
            na=False,
        )
        |
        countries.str.contains(
            r"(^|,)\s*vietnam\s*(,|$)",
            regex=True,
            na=False,
        )
    )

    selected = chunk[is_vietnam].copy()

    if not selected.empty:
        vietnam_parts.append(selected)

if not vietnam_parts:
    print("Khong tim thay du lieu Viet Nam.")
    raise SystemExit

df = pd.concat(vietnam_parts, ignore_index=True)

# Kiểm tra điều kiện hợp lệ
df["ingredients_text"] = (
    df["ingredients_text"].fillna("").str.strip()
)

df["nova_group"] = pd.to_numeric(
    df["nova_group"],
    errors="coerce",
)

has_ingredients = df["ingredients_text"].str.len() > 0
valid_nova = df["nova_group"].isin([1, 2, 3, 4])

valid = df[has_ingredients & valid_nova].copy()

# Kiểm tra trùng lặp
unique_valid = valid.drop_duplicates(
    subset=["ingredients_text", "nova_group"]
)

print("\n===== KET QUA KIEM TRA VIET NAM =====")
print("Tong dong co thong tin Viet Nam:", len(df))
print("Co nguyen lieu:", int(has_ingredients.sum()))
print("Co nhan NOVA hop le:", int(valid_nova.sum()))
print("Thoa ca hai dieu kien:", len(valid))
print("Sau khi loai trung:", len(unique_valid))

print("\nPhan bo NOVA trong du lieu hop le:")
print(
    unique_valid["nova_group"]
    .value_counts()
    .sort_index()
)

print("\nSo dong thieu nguyen lieu:", int((~has_ingredients).sum()))
print("So dong thieu/khong hop le nhan NOVA:", int((~valid_nova).sum()))
