import pandas as pd
from pathlib import Path


def clean_power_value(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in ["[N/A]", "N/A", ""]:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def analyze_log(log_path):
    df = pd.read_csv(log_path)

    df["power_w_clean"] = df["power_w"].apply(clean_power_value)
    df["gpu_util_percent"] = pd.to_numeric(df["gpu_util_percent"], errors="coerce")
    df["memory_used_mb"] = pd.to_numeric(df["memory_used_mb"], errors="coerce")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")

    power_available = df["power_w_clean"].notna().any()

    summary = {
        "gpu_log_file": str(log_path).replace("\\", "/"),
        "num_samples_log": len(df),
        "power_available": power_available,
        "avg_power_w": df["power_w_clean"].mean(),
        "max_power_w": df["power_w_clean"].max(),
        "min_power_w": df["power_w_clean"].min(),
        "avg_gpu_util_percent": df["gpu_util_percent"].mean(),
        "max_gpu_util_percent": df["gpu_util_percent"].max(),
        "avg_memory_used_mb": df["memory_used_mb"].mean(),
        "max_memory_used_mb": df["memory_used_mb"].max(),
        "avg_temperature_c": df["temperature_c"].mean(),
        "max_temperature_c": df["temperature_c"].max(),
    }

    return summary


def main():
    experiments_path = Path("results/experiments_with_logs.csv")
    output_path = Path("results/log_summary.csv")

    experiments = pd.read_csv(experiments_path)

    summaries = []

    for log_file in experiments["gpu_log_file"]:
        log_path = Path(log_file)

        if log_path.exists():
            summaries.append(analyze_log(log_path))
        else:
            summaries.append({
                "gpu_log_file": str(log_path).replace("\\", "/"),
                "num_samples_log": 0,
                "power_available": False,
                "avg_power_w": None,
                "max_power_w": None,
                "min_power_w": None,
                "avg_gpu_util_percent": None,
                "max_gpu_util_percent": None,
                "avg_memory_used_mb": None,
                "max_memory_used_mb": None,
                "avg_temperature_c": None,
                "max_temperature_c": None,
            })

    summary_df = pd.DataFrame(summaries)

    final_df = experiments.merge(summary_df, on="gpu_log_file", how="left")

    # Energia = potência média durante o treino x tempo de treino.
    # 1 Wh = 3600 J. Fica vazio quando a GPU não informa a potência.
    final_df["energy_j"] = final_df["avg_power_w"] * final_df["train_time_seconds"]
    final_df["energy_wh"] = final_df["energy_j"] / 3600

    Path("results").mkdir(exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"Resumo salvo em: {output_path}")
    print(final_df[["architecture", "batch_size", "image_size", "train_time_seconds", "avg_power_w", "energy_j"]])


if __name__ == "__main__":
    main()