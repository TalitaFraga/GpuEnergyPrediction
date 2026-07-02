import argparse
import csv
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
    with torch.cuda.device(0) if torch.cuda.is_available() else torch.device("cpu"):
        macs, params = get_model_complexity_info(
            model,
            (3, image_size, image_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
    return macs * 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--dataset_size", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    architecture = "resnet18"
    dataset_name = "cifar10"
    optimizer_name = "sgd"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

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
    model_size_mb = (params * 4) / (1024 ** 2)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

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
    train_time = end - start

    Path("results").mkdir(exist_ok=True)
    file_path = Path("results/experiments.csv")
    file_exists = file_path.exists()

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
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
                "layers",
                "model_size_mb",
                "device",
                "gpu_count",
                "gpu_name",
                "train_time_seconds"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            architecture,
            dataset_name,
            optimizer_name,
            args.batch_size,
            args.epochs,
            args.dataset_size,
            args.image_size,
            args.num_workers,
            params,
            flops,
            layers,
            round(model_size_mb, 4),
            device,
            gpu_count,
            gpu_name,
            round(train_time, 4)
        ])

    print("Treinamento finalizado.")
    print(f"Tempo total: {train_time:.2f} segundos")
    print(f"Parâmetros: {params}")
    print(f"FLOPs: {flops}")


if __name__ == "__main__":
    main()