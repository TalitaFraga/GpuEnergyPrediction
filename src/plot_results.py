from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def create_output_folder():
    Path("results/figures").mkdir(parents=True, exist_ok=True)


def plot_training_time(df):
    plt.figure(figsize=(7, 5))

    plt.plot(
        df["batch_size"],
        df["train_time_seconds"],
        marker="o",
        linewidth=2
    )

    plt.title("Batch Size vs Training Time")
    plt.xlabel("Batch Size")
    plt.ylabel("Training Time (seconds)")
    plt.grid(True)

    plt.savefig(
        "results/figures/batch_vs_training_time.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_temperature(df):
    plt.figure(figsize=(7, 5))

    plt.plot(
        df["batch_size"],
        df["avg_temperature_c"],
        marker="o",
        linewidth=2
    )

    plt.title("Batch Size vs Average GPU Temperature")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Temperature (°C)")
    plt.grid(True)

    plt.savefig(
        "results/figures/batch_vs_temperature.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_gpu_utilization(df):
    plt.figure(figsize=(7, 5))

    plt.plot(
        df["batch_size"],
        df["avg_gpu_util_percent"],
        marker="o",
        linewidth=2
    )

    plt.title("Batch Size vs GPU Utilization")
    plt.xlabel("Batch Size")
    plt.ylabel("Average GPU Utilization (%)")
    plt.grid(True)

    plt.savefig(
        "results/figures/batch_vs_gpu_utilization.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_memory(df):
    plt.figure(figsize=(7, 5))

    plt.plot(
        df["batch_size"],
        df["avg_memory_used_mb"],
        marker="o",
        linewidth=2
    )

    plt.title("Batch Size vs GPU Memory Usage")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Memory (MB)")
    plt.grid(True)

    plt.savefig(
        "results/figures/batch_vs_memory.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_power(df):
    if not df["power_available"].any():
        print("Nenhum experimento possui dados de potência. Gráfico ignorado.")
        return

    df_power = df[df["power_available"]]

    plt.figure(figsize=(7, 5))

    plt.plot(
        df_power["batch_size"],
        df_power["avg_power_w"],
        marker="o",
        linewidth=2
    )

    plt.title("Batch Size vs Average GPU Power")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Power (W)")
    plt.grid(True)

    plt.savefig(
        "results/figures/batch_vs_power.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    create_output_folder()

    df = pd.read_csv("results/log_summary.csv")

    df = df.sort_values("batch_size")

    plot_training_time(df)
    plot_temperature(df)
    plot_gpu_utilization(df)
    plot_memory(df)
    plot_power(df)

    print("\nGráficos gerados em:")
    print("results/figures/")


if __name__ == "__main__":
    main()