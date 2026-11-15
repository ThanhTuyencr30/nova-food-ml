from pathlib import Path
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT_DIR = Path(__file__).resolve().parents[3]
PREDICTIONS_PATH = ROOT_DIR / "reports" / "test_predictions.csv"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

LABELS = [1, 2, 3, 4]


def evaluate(predictions_path=PREDICTIONS_PATH):
    predictions_path = Path(predictions_path)

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dự đoán: {predictions_path}"
        )

    df = pd.read_csv(predictions_path)

    required = {"y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError("File phải có hai cột y_true và y_pred.")

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        REPORT_DIR / "classification_report.csv",
        encoding="utf-8-sig",
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=[f"NOVA {i}" for i in LABELS],
        yticklabels=[f"NOVA {i}" for i in LABELS],
        xlabel="Nhãn dự đoán",
        ylabel="Nhãn thực tế",
        title="Confusion Matrix - NOVA Classification",
    )

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Metrics summary
    metrics = {
        "samples": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    (REPORT_DIR / "evaluation_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== KẾT QUẢ ĐÁNH GIÁ ===")
    print(f"Số mẫu kiểm tra: {metrics['samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")

    print("\n=== CLASSIFICATION REPORT ===")
    print(report_df.round(4).to_string())

    print("\nĐã tạo:")
    print(REPORT_DIR / "classification_report.csv")
    print(REPORT_DIR / "evaluation_summary.json")
    print(FIGURE_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=str(PREDICTIONS_PATH),
    )
    args = parser.parse_args()

    evaluate(args.predictions)