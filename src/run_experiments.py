import argparse
import itertools
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", nargs="+", default=["resnet18", "mobilenet_v2", "resnet50", "vgg11"])
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--image_sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--dataset_sizes", nargs="+", type=int, default=[2000])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="teste rápido: poucas configurações")
    args = parser.parse_args()

    if args.quick:
        args.architectures = ["resnet18", "mobilenet_v2"]
        args.batch_sizes = [16, 32]
        args.image_sizes = [32]
        args.dataset_sizes = [500]
        args.repetitions = 1

    configs = list(itertools.product(
        args.architectures,
        args.batch_sizes,
        args.image_sizes,
        args.dataset_sizes,
        range(1, args.repetitions + 1)
    ))

    failures = []

    for i, (arch, batch, image_size, dataset_size, rep) in enumerate(configs, start=1):
        print(
            f"\n[{i}/{len(configs)}] {arch} | batch_size={batch} | "
            f"image_size={image_size} | dataset_size={dataset_size} | repetição {rep}"
        )

        result = subprocess.run([
            sys.executable,
            "src/train_with_logger.py",
            "--architecture", arch,
            "--batch_size", str(batch),
            "--epochs", str(args.epochs),
            "--dataset_size", str(dataset_size),
            "--image_size", str(image_size)
        ])

        # Configurações grandes podem estourar a memória da GPU; seguimos com as demais.
        if result.returncode != 0:
            failures.append((arch, batch, image_size, dataset_size, rep))

    print(f"\nExperimentos concluídos: {len(configs) - len(failures)}/{len(configs)}")

    if failures:
        print("Falharam:")
        for arch, batch, image_size, dataset_size, rep in failures:
            print(f"  {arch} | batch_size={batch} | image_size={image_size} | dataset_size={dataset_size} | rep {rep}")


if __name__ == "__main__":
    main()
