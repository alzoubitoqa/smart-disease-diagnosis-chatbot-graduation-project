from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_history(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_plot(save_dir: Path, filename: str) -> None:
    plt.tight_layout()
    plt.savefig(save_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_curve(history: dict, model_name: str, figure_num: str, save_name: str, save_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.get("accuracy", []), label="Train Accuracy")
    plt.plot(history.get("val_accuracy", []), label="Validation Accuracy")
    plt.title(f"Figure {figure_num}: {model_name} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(save_dir, save_name)


def plot_loss_curve(history: dict, model_name: str, figure_num: str, save_name: str, save_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.get("loss", []), label="Train Loss")
    plt.plot(history.get("val_loss", []), label="Validation Loss")
    plt.title(f"Figure {figure_num}: {model_name} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(save_dir, save_name)


def plot_combined_accuracy(histories: dict, save_dir: Path) -> None:
    plt.figure(figsize=(9, 6))
    for model_name, history in histories.items():
        plt.plot(history.get("val_accuracy", []), label=f"{model_name} Val Acc")
    plt.title("Figure 4.8: Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(save_dir, "Figure_4.8_Validation_Accuracy_Comparison")


def plot_combined_loss(histories: dict, save_dir: Path) -> None:
    plt.figure(figsize=(9, 6))
    for model_name, history in histories.items():
        plt.plot(history.get("val_loss", []), label=f"{model_name} Val Loss")
    plt.title("Figure 4.9: Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(save_dir, "Figure_4.9_Validation_Loss_Comparison")


def plot_final_comparison(csv_path: Path, save_dir: Path) -> None:
    df = pd.read_csv(csv_path)

    # Figure 4.10 - Accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["accuracy"])
    plt.title("Figure 4.10: Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(save_dir, "Figure_4.10_Model_Accuracy_Comparison")

    # Figure 4.11 - F1-score
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["f1_macro"])
    plt.title("Figure 4.11: Model F1-score Comparison")
    plt.xlabel("Model")
    plt.ylabel("F1-score")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(save_dir, "Figure_4.11_Model_F1_Score_Comparison")

    # Figure 4.12 - Precision
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["precision_macro"])
    plt.title("Figure 4.12: Model Precision Comparison")
    plt.xlabel("Model")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(save_dir, "Figure_4.12_Model_Precision_Comparison")

    # Figure 4.13 - Recall
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df["recall_macro"])
    plt.title("Figure 4.13: Model Recall Comparison")
    plt.xlabel("Model")
    plt.ylabel("Recall")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    save_plot(save_dir, "Figure_4.13_Model_Recall_Comparison")


def plot_all_results(
    rnn_history_path: str,
    vanilla_lstm_history_path: str,
    deep_lstm_history_path: str,
    comparison_csv_path: str,
    output_dir: str = "outputs/figures"
) -> None:
    save_dir = Path(output_dir)
    ensure_dir(save_dir)

    rnn_history = load_history(Path(rnn_history_path))
    vanilla_history = load_history(Path(vanilla_lstm_history_path))
    deep_history = load_history(Path(deep_lstm_history_path))

    histories = {
        "Simple RNN": rnn_history,
        "Vanilla LSTM": vanilla_history,
        "Deep LSTM": deep_history,
    }

    # 4.8 - 4.13
    plot_combined_accuracy(histories, save_dir)
    plot_combined_loss(histories, save_dir)
    plot_final_comparison(Path(comparison_csv_path), save_dir)

    # 4.14 - 4.19
    plot_accuracy_curve(
        rnn_history,
        "Simple RNN",
        "4.14",
        "Figure_4.14_Simple_RNN_Accuracy_Curve",
        save_dir,
    )
    plot_loss_curve(
        rnn_history,
        "Simple RNN",
        "4.15",
        "Figure_4.15_Simple_RNN_Loss_Curve",
        save_dir,
    )

    plot_accuracy_curve(
        vanilla_history,
        "Vanilla LSTM",
        "4.16",
        "Figure_4.16_Vanilla_LSTM_Accuracy_Curve",
        save_dir,
    )
    plot_loss_curve(
        vanilla_history,
        "Vanilla LSTM",
        "4.17",
        "Figure_4.17_Vanilla_LSTM_Loss_Curve",
        save_dir,
    )

    plot_accuracy_curve(
        deep_history,
        "Deep LSTM",
        "4.18",
        "Figure_4.18_Deep_LSTM_Accuracy_Curve",
        save_dir,
    )
    plot_loss_curve(
        deep_history,
        "Deep LSTM",
        "4.19",
        "Figure_4.19_Deep_LSTM_Loss_Curve",
        save_dir,
    )

    print(f"All plots saved successfully in: {save_dir}")


if __name__ == "__main__":
    plot_all_results(
        rnn_history_path="artifacts/reports/rnn_history.json",
        vanilla_lstm_history_path="artifacts/reports/vanilla_lstm_history.json",
        deep_lstm_history_path="artifacts/reports/deep_lstm_history.json",
        comparison_csv_path="artifacts/reports/final_comparison.csv",
        output_dir="outputs/figures"
    )