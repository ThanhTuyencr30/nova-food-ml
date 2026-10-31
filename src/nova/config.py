from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "off_nova.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "off_nova.csv"
VIETNAM_DATA_PATH = ROOT_DIR / "data" / "processed" / "vietnam.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

TEXT_COLUMN = "ingredients_text"
TARGET_COLUMN = "nova_group"
RANDOM_STATE = 42
TEST_SIZE = 0.2