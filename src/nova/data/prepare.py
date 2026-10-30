import pandas as pd
from src.nova.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TEXT_COLUMN, TARGET_COLUMN

REQUIRED_COLUMNS = [TEXT_COLUMN, TARGET_COLUMN]

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

def main():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {RAW_DATA_PATH}. Hãy đặt dữ liệu nguồn vào data/raw/off_nova.csv"
        )
    df = pd.read_csv(RAW_DATA_PATH)
    cleaned = prepare_dataframe(df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Đã lưu {len(cleaned)} dòng vào {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    main()
