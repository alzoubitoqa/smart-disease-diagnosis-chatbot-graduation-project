from src.evaluation.plots import plot_all_results

if __name__ == "__main__":
    plot_all_results(
    rnn_history_path="artifacts/reports/rnn_history.json",
    vanilla_lstm_history_path="artifacts/reports/vanilla_lstm_history.json",
    deep_lstm_history_path="artifacts/reports/deep_lstm_history.json",
    comparison_csv_path="artifacts/reports/final_comparison.csv",
    output_dir="outputs/figures"
)