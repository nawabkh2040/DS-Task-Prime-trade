import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def main():
    data_dir = Path("data")
    sent_fp = data_dir / "sentiment_summary_by_daily.csv"
    merged_fp = data_dir / "merged_by_date.csv"

    if not sent_fp.exists() or not merged_fp.exists():
        print(f"Missing files: {sent_fp.exists()} {merged_fp.exists()}")
        return

    sent = pd.read_csv(sent_fp)
    merged = pd.read_csv(merged_fp, parse_dates=["date"]) if merged_fp.exists() else pd.DataFrame()

    # Order sentiments by mean_total_pnl
    order = sent.sort_values("mean_total_pnl", ascending=False)["classification"].tolist()

    sns.set(style="whitegrid")

    # Bar plot: mean total pnl per classification
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=sent, x="classification", y="mean_total_pnl", order=order, palette="viridis")
    ax.set_title("Mean Total PnL by Sentiment Classification")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out1 = data_dir / "fig_mean_total_pnl_by_sentiment.png"
    plt.savefig(out1, dpi=150)
    print("Saved", out1)
    plt.close()

    # Boxplot: distribution of daily total_pnl by classification
    if not merged.empty and "total_pnl" in merged.columns:
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=merged, x="classification", y="total_pnl", order=order)
        # zoom to central range to reduce extreme outlier stretch
        lower = merged["total_pnl"].quantile(0.01)
        upper = merged["total_pnl"].quantile(0.99)
        try:
            ax.set_ylim(lower, upper)
        except Exception:
            pass
        ax.set_title("Distribution of Daily Total PnL by Sentiment (boxplot)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out2 = data_dir / "fig_daily_total_pnl_box_by_sentiment.png"
        plt.savefig(out2, dpi=150)
        print("Saved", out2)
        plt.close()
    else:
        print("Merged daily file missing or lacks 'total_pnl'; skipping boxplot.")


if __name__ == "__main__":
    main()
