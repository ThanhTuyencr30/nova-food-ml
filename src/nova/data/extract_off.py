import pandas as pd

from src.nova.config import ROOT_DIR, RAW_DATA_PATH

DUMP_PATH = ROOT_DIR / "data" / "raw" / "en.openfoodfacts.org.products.csv.gz"
COLUMNS = [
    "product_name", "countries_tags", "ingredients_text",
    "nova_group", "sugars_100g", "salt_100g",
]
SAMPLE_FRAC = 0.1      # chỉ giữ 10% sản phẩm ngoài Việt Nam cho dữ liệu gọn
CHUNK_SIZE = 200_000   # đọc từng khúc, không nạp hết vào RAM


def main():
    if not DUMP_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DUMP_PATH}. Hãy tải file .gz từ Open Food Facts vào data/raw/"
        )

    pieces, total = [], 0
    reader = pd.read_csv(
        DUMP_PATH, sep="\t", usecols=COLUMNS, chunksize=CHUNK_SIZE,
        low_memory=False, on_bad_lines="skip",
    )
    try:
        for chunk in reader:
            total += len(chunk)
            chunk["nova_group"] = pd.to_numeric(chunk["nova_group"], errors="coerce")
            chunk = chunk.dropna(subset=["nova_group", "ingredients_text"]).copy()
            chunk = chunk[chunk["nova_group"].isin([1, 2, 3, 4])].copy()
            chunk["is_vietnam"] = (
                chunk["countries_tags"].fillna("").str.contains("en:vietnam").astype(int)
            )
            vn = chunk[chunk["is_vietnam"] == 1]
            other = chunk[chunk["is_vietnam"] == 0].sample(frac=SAMPLE_FRAC, random_state=42)
            pieces.append(pd.concat([vn, other]))
            print("da doc", total, "dong")
    except EOFError:
        print("CANH BAO: file .gz chua tai du, chi dung phan doc duoc:", total, "dong")

    if not pieces:
        raise SystemExit("Khong doc duoc dong nao. Hay tai lai file .gz.")

    df = pd.concat(pieces).drop(columns=["countries_tags"])
    df["nova_group"] = df["nova_group"].astype(int)
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    print(df.shape)
    print(df["nova_group"].value_counts().sort_index())
    print("So san pham Viet Nam:", int(df["is_vietnam"].sum()))


if __name__ == "__main__":
    main()