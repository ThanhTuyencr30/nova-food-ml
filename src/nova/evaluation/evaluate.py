import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

def evaluate_model(model, X_test, y_test, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = model.predict(X_test)

    macro_f1 = f1_score(y_test, predictions, average="macro")
    report = classification_report(y_test, predictions, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=[1, 2, 3, 4], values_format="d"
    )
    display.ax_.set_title("Confusion Matrix - NOVA")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()
    return macro_f1, report
