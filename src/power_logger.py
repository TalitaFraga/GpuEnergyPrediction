import csv
import subprocess
import time
from pathlib import Path


def log_gpu_power(output_file, interval=0.1):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "power_w",
            "gpu_util_percent",
            "memory_used_mb",
            "temperature_c"
        ])

        print("Iniciando coleta de energia... (Ctrl+C para parar)")

        while True:
            try:
                result = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ]).decode("utf-8").strip()

                row = [x.strip() for x in result.split(",")]

                writer.writerow(row)
                f.flush()

                time.sleep(interval)

            except KeyboardInterrupt:
                print("\nColeta encerrada.")
                break


if __name__ == "__main__":
    log_gpu_power("logs/gpu_power.csv")