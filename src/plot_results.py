from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("train_time_seconds", "Training Time (s)", "batch_vs_training_time.png"),
    ("avg_temperature_c", "Average Temperature (°C)", "batch_vs_temperature.png"),
    ("avg_gpu_util_percent", "Average GPU Utilization (%)", "batch_vs_gpu_utilization.png"),
    ("max_memory_used_mb", "Peak GPU Memory (MB)", "batch_vs_memory.png"),
    ("avg_power_w", "Average Power (W)", "batch_vs_power.png"),
    ("energy_j", "Energy (J)", "batch_vs_energy.png"),
]


def create_output_folder():
    Path("results/figures").mkdir(parents=True, exist_ok=True)


def plot_metric(df, column, ylabel, filename):
    if df[column].isna().all():
        print(f"Sem dados para '{column}'. Gráfico ignorado.")
        return

    # Média das repetições de cada configuração.
    grouped = (
        df.groupby(["image_size", "architecture", "batch_size"])[column]
        .mean()
        .reset_index()
    )

    image_sizes = sorted(grouped["image_size"].unique())

    fig, axes = plt.subplots(
        1, len(image_sizes),
        figsize=(6 * len(image_sizes), 5),
        squeeze=False
    )

    for ax, image_size in zip(axes[0], image_sizes):
        subset = grouped[grouped["image_size"] == image_size]

        for architecture, arch_df in subset.groupby("architecture"):
            arch_df = arch_df.sort_values("batch_size")
            ax.plot(
                arch_df["batch_size"],
                arch_df[column],
                marker="o",
                linewidth=2,
                label=architecture
            )

        ax.set_title(f"Image size {image_size}x{image_size}")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    fig.suptitle(f"Batch Size vs {ylabel}")

    fig.savefig(
        f"results/figures/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():

    create_output_folder()

    df = pd.read_csv("results/log_summary.csv")

    for column, ylabel, filename in METRICS:
        plot_metric(df, column, ylabel, filename)

    print("\nGráficos gerados em:")
    print("results/figures/")


if __name__ == "__main__":
    main()
