import argparse
import csv
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from ptflops import get_model_complexity_info


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_layers(model):
    return sum(1 for _ in model.modules())


def calculate_flops(model, image_size):
    macs, _ = get_model_complexity_info(
        model,
        (3, image_size, image_size),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False
    )
    return macs * 2


def gpu_logger(output_file, stop_event, interval=0.5):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "power_w",
            "gpu_util_percent",
            "memory_used_mb",
            "temperature_c",
            "logger_status"
        ])

        while not stop_event.is_set():
            try:
                result = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits"
                ], stderr=subprocess.DEVNULL).decode("utf-8").strip()

                row = [x.strip() for x in result.split(",")]
                writer.writerow(row + ["ok"])

            except Exception:
                writer.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "error"
                ])

            f.flush()
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    args = parser.parse_args()

    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda
    log_power_available = "unknown"

    print(f"Experimento: {experiment_id}")
    print(f"Dispositivo: {device}")
    print(f"GPU: {gpu_name}")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    dataset = Subset(dataset, range(args.dataset_size))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    model = models.resnet18(weights=None, num_classes=10)

    params = count_params(model)
    layers = count_layers(model)
    flops = calculate_flops(model, args.image_size)
    gflops = flops / 1e9
    model_size_mb = (params * 4) / (1024 ** 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    log_file = f"logs/gpu_log_{experiment_id}.csv"

    stop_event = threading.Event()
    logger_thread = threading.Thread(
        target=gpu_logger,
        args=(log_file, stop_event)
    )

    logger_thread.start()

    start = time.time()

    for epoch in range(args.epochs):
        total_loss = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Época {epoch + 1}/{args.epochs} | Loss: {total_loss:.4f}")

    end = time.time()

    stop_event.set()
    logger_thread.join()

    train_time = end - start

    Path("results").mkdir(exist_ok=True)
    file_path = Path("results/experiments_with_logs.csv")
    file_exists = file_path.exists()

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "experiment_id",
                "timestamp",
                "architecture",
                "dataset",
                "optimizer",
                "batch_size",
                "epochs",
                "dataset_size",
                "image_size",
                "num_workers",
                "params",
                "flops",
                "gflops",
                "layers",
                "model_size_mb",
                "learning_rate",
                "device",
                "gpu_count",
                "gpu_name",
                "torch_version",
                "cuda_available",
                "cuda_version",
                "log_power_available",
                "train_time_seconds",
                "gpu_log_file"
            ])

        writer.writerow([
            experiment_id,
            datetime.now().isoformat(timespec="seconds"),
            "resnet18",
            "cifar10",
            "sgd",
            args.batch_size,
            args.epochs,
            args.dataset_size,
            args.image_size,
            args.num_workers,
            params,
            flops,
            round(gflops, 6),
            layers,
            round(model_size_mb, 4),
            args.learning_rate,
            device,
            gpu_count,
            gpu_name,
            torch_version,
            cuda_available,
            cuda_version,
            log_power_available,
            round(train_time, 4),
            log_file
        ])

    print("Treinamento finalizado.")
    print(f"Tempo total: {train_time:.2f} segundos")
    print(f"Log da GPU salvo em: {log_file}")


if __name__ == "__main__":
    main()