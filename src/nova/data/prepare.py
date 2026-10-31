import pandas as pd
from src.nova.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, VIETNAM_DATA_PATH, TEXT_COLUMN, TARGET_COLUMN
)

REQUIRED_COLUMNS = [TEXT_COLUMN, TARGET_COLUMN]
VIETNAM_COLUMN = "is_vietnam"
VIET_CHARS = "[ăđơư\u1ea0-\u1ef9]"  # ký tự đặc trưng tiếng Việt


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    result = df.copy()
    result[TEXT_COLUMN] = result[TEXT_COLUMN].fillna("").astype(str).str.strip()
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
    result = result.dropna(subset=[TARGET_COLUMN])
    result = result[result[TARGET_COLUMN].isin([1, 2, 3, 4])]
    result = result[result[TEXT_COLUMN] != ""]
    result[TARGET_COLUMN] = result[TARGET_COLUMN].astype(int)
    return result.drop_duplicates().reset_index(drop=True)


def add_vietnamese_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột co_chu_viet = 1 nếu thành phần có chữ tiếng Việt (có dấu)."""
    result = df.copy()
    result["co_chu_viet"] = (
        result[TEXT_COLUMN].str.contains(VIET_CHARS, case=False, regex=True).astype(int)
    )
    return result


def split_vietnam(df: pd.DataFrame):
    """Trả về (sản phẩm ngoài Việt Nam, sản phẩm Việt Nam)."""
    if VIETNAM_COLUMN not in df.columns:
        return df.reset_index(drop=True), df.iloc[0:0]
    vietnam = df[df[VIETNAM_COLUMN] == 1].reset_index(drop=True)
    others = df[df[VIETNAM_COLUMN] == 0].reset_index(drop=True)
    return others, vietnam


def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {RAW_DATA_PATH}. Hãy đặt dữ liệu nguồn vào data/raw/off_nova.csv"
        )
    df = pd.read_csv(RAW_DATA_PATH)
    cleaned = add_vietnamese_flag(prepare_dataframe(df))
    others, vietnam = split_vietnam(cleaned)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    others.to_csv(PROCESSED_DATA_PATH, index=False)
    vietnam.to_csv(VIETNAM_DATA_PATH, index=False)

    print(f"Đã lưu {len(others)} dòng (ngoài Việt Nam) vào {PROCESSED_DATA_PATH}")
    print(f"Đã lưu {len(vietnam)} dòng (Việt Nam) vào {VIETNAM_DATA_PATH}")
    if len(vietnam):
        print("Trong đó có chữ tiếng Việt:", int(vietnam["co_chu_viet"].sum()))


if __name__ == "__main__":
    main()