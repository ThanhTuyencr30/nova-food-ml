
import csv
import pandas as pd

from src.nova.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    TEXT_COLUMN,
    TARGET_COLUMN,
)

CHUNKSIZE = 10000

REQUIRED_COLUMNS = [
    TEXT_COLUMN,
    TARGET_COLUMN,
]

OPTIONAL_COLUMNS = [
    "product_name",
    "countries_tags",
    "countries_en",
]


def clean_text(value):
    if pd.isna(value):
        return ""

    return " ".join(str(value).strip().split())


def prepare_dataframe(df):
    # Kiểm tra các cột bắt buộc
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Thiếu các cột bắt buộc: {missing}"
        )

    # Làm sạch nguyên liệu
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

    # Chuyển nhãn NOVA thành số
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce"
    )

    # Chỉ giữ NOVA 1, 2, 3, 4
    df = df[
        df[TARGET_COLUMN].isin([1, 2, 3, 4])
    ].copy()

    # Loại bỏ nguyên liệu rỗng
    df = df[
        df[TEXT_COLUMN].str.len() > 0
    ].copy()

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN].astype("int8")
    )

    # Loại bỏ bản ghi trùng
    df = df.drop_duplicates(
        subset=[TEXT_COLUMN, TARGET_COLUMN]
    )

    return df


def add_vietnam_flag(df):
    df = df.copy()

    text_columns = []
    for col in [TEXT_COLUMN, "product_name", "countries_tags", "countries_en"]:
        if col in df.columns:
            text_columns.append(df[col].fillna("").astype(str))

    if text_columns:
        combined_text = text_columns[0]
        for col_text in text_columns[1:]:
            combined_text = combined_text + " " + col_text
    else:
        combined_text = pd.Series("", index=df.index)

    df["co_chu_viet"] = combined_text.str.contains(
        r"[à-ỹÀ-ỸăắằẳẵặấầẩẫậéèẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựýỳỷỹỵđĐ]",
        regex=True,
        na=False,
    ).astype("int8")

    countries_tags = (
        df["countries_tags"]
        if "countries_tags" in df.columns
        else pd.Series("", index=df.index)
    )

    countries_en = (
        df["countries_en"]
        if "countries_en" in df.columns
        else pd.Series("", index=df.index)
    )

    countries = (
        countries_tags.fillna("")
        + ","
        + countries_en.fillna("")
    ).str.lower()

    if "is_vietnam" in df.columns:
        df["is_vietnam"] = pd.to_numeric(
            df["is_vietnam"], errors="coerce"
        ).fillna(0).astype("int8")
    else:
        df["is_vietnam"] = countries.str.contains(
            r"vietnam|việt nam",
            regex=True,
            na=False,
        ).astype("int8")

    return df


def add_vietnamese_flag(df):
    return add_vietnam_flag(df)


def split_vietnam(df):
    df = df.copy()
    if "co_chu_viet" not in df.columns:
        df = add_vietnamese_flag(df)

    vietnam = df[df["co_chu_viet"] == 1].copy()
    others = df[df["co_chu_viet"] == 0].copy()
    return others, vietnam


def main():
    print("Đang đọc dữ liệu nguồn...")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {RAW_DATA_PATH}"
        )

    RAW_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    reader = pd.read_csv(
        RAW_DATA_PATH,
        sep="\t",
        encoding="utf-8-sig",
        dtype=str,
        engine="c",
        quoting=csv.QUOTE_NONE,
        chunksize=CHUNKSIZE,
        usecols=lambda col: (
            col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        ),
        on_bad_lines="warn",
    )

    processed_chunks = []
    total_rows = 0

    for chunk in reader:
        total_rows += len(chunk)

        chunk = prepare_dataframe(chunk)
        chunk = add_vietnam_flag(chunk)

        if not chunk.empty:
            processed_chunks.append(chunk)

        print(
            f"Đã đọc {total_rows:,} dòng nguồn..."
        )

    if not processed_chunks:
        raise ValueError(
            "Không còn dữ liệu hợp lệ sau khi làm sạch."
        )

    result = pd.concat(
        processed_chunks,
        ignore_index=True
    )

    result = result.drop_duplicates(
        subset=[TEXT_COLUMN, TARGET_COLUMN]
    )

    # Lưu toàn bộ dữ liệu đã xử lý
    result.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Tách dữ liệu có thông tin quốc gia Việt Nam
    vietnam_path = (
        PROCESSED_DATA_PATH.parent
        / "off_nova_vietnam.csv"
    )

    vietnam = result[
        result["is_vietnam"] == 1
    ].copy()

    vietnam.to_csv(
        vietnam_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== KẾT QUẢ TIỀN XỬ LÝ =====")
    print(f"Tổng dòng nguồn đã đọc: {total_rows:,}")
    print(f"Tổng sản phẩm hợp lệ: {len(result):,}")
    print(f"Dữ liệu Việt Nam: {len(vietnam):,}")
    print(f"Dữ liệu quốc tế/không xác định: {len(result) - len(vietnam):,}")
    print("\nPhân bố nhãn NOVA:")
    print(result[TARGET_COLUMN].value_counts().sort_index())
    print(f"\nĐã lưu: {PROCESSED_DATA_PATH}")
    print(f"Đã lưu: {vietnam_path}")


if __name__ == "__main__":
    main()
