from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from ultralytics import YOLO


def parse_batch(value: str) -> int | str:
    if value.lower() == "auto":
        return -1
    return int(value)


def parse_cache(value: str) -> bool | str:
    normalized = value.strip().lower()
    if normalized in {"false", "off", "none", "0"}:
        return False
    if normalized in {"true", "on", "1", "ram"}:
        return True
    if normalized == "disk":
        return "disk"
    raise argparse.ArgumentTypeError("cache must be one of: false, ram, disk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 model for helmet/no_helmet detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to YOLO dataset yaml file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base YOLOv8 model checkpoint. Defaults to YOLOv8s for a stronger accuracy/speed balance.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument(
        "--batch",
        type=parse_batch,
        default=4,
        help="Batch size. Use a small fixed value for stable training on low-memory GPUs.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="models",
        help="Directory where training outputs will be stored.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="helmet_detector",
        help="Experiment name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device id or cpu. Defaults to automatic selection.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dataloader workers. Use 2 for a good speed/stability balance on Windows.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision if the GPU supports it safely.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the checkpoint passed in --model.",
    )
    parser.add_argument(
        "--cache",
        type=parse_cache,
        default="disk",
        help="Dataset caching mode: false, ram, or disk.",
    )
    return parser.parse_args()


def save_metrics(metrics: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_path


def is_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    keywords = (
        "out of memory",
        "cublas_status_not_initialized",
        "cuda error",
        "arraymemoryerror",
        "insufficient memory",
        "cudnn_status_internal_error_host_allocation_failed",
        "host allocation failed",
    )
    return any(keyword in message for keyword in keywords)


def choose_training_defaults(args: argparse.Namespace) -> tuple[int | str, int, bool | str]:
    batch = args.batch
    workers = args.workers
    cache = args.cache

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0).lower()
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb <= 4.5:
            if batch == -1 or batch > 2:
                batch = 2
            workers = min(workers, 0)
            if cache == "disk":
                cache = False
        elif total_vram_gb <= 6:
            if batch == -1:
                batch = 4
            if workers <= 0:
                workers = 2
        if "gtx 1650" in device_name:
            if batch == -1 or batch > 2:
                batch = 2
            workers = 0
            if cache == "disk":
                cache = False

    return batch, workers, cache


def get_completed_epochs(checkpoint_path: Path) -> int:
    run_dir = checkpoint_path.parent.parent
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return 0

    lines = [line for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def run_training(args: argparse.Namespace):
    batch, workers, cache = choose_training_defaults(args)
    effective_epochs = args.epochs
    model_path = args.model
    run_name = args.name

    if args.resume:
        checkpoint_path = Path(args.model)
        completed_epochs = get_completed_epochs(checkpoint_path)
        remaining_epochs = args.epochs - completed_epochs
        if remaining_epochs <= 0:
            raise ValueError(
                f"Checkpoint already has {completed_epochs} completed epochs, which meets or exceeds the requested {args.epochs} epochs."
            )
        effective_epochs = remaining_epochs
        model_path = str(checkpoint_path)
        if args.name == "helmet_detector":
            run_name = f"{checkpoint_path.parent.parent.name}_resume"
        print(
            f"Safe resume mode: completed_epochs={completed_epochs}, "
            f"remaining_epochs={remaining_epochs}, batch={batch}, workers={workers}, cache={cache}"
        )

    model = YOLO(model_path)

    attempts: list[tuple[int | str, int, bool | str]] = [(batch, workers, cache)]
    if isinstance(batch, int) and batch > 2:
        attempts.append((max(2, batch // 2), 0, False))
    if (2, 0, False) not in attempts:
        attempts.append((2, 0, False))
    if (1, 0, False) not in attempts:
        attempts.append((1, 0, False))

    last_error: Exception | None = None
    for index, (attempt_batch, attempt_workers, attempt_cache) in enumerate(attempts, start=1):
        try:
            print(
                f"Training attempt {index}: batch={attempt_batch}, workers={attempt_workers}, "
                f"cache={attempt_cache}, device={args.device or 'auto'}, resume={bool(args.resume)}"
            )
            return model.train(
                data=args.data,
                epochs=effective_epochs,
                imgsz=args.imgsz,
                batch=attempt_batch,
                project=args.project,
                name=run_name,
                device=args.device,
                workers=attempt_workers,
                amp=args.amp,
                pretrained=not args.resume,
                resume=False,
                cache=attempt_cache,
                deterministic=False,
                plots=False,
                verbose=True,
            )
        except Exception as exc:
            last_error = exc
            if not is_memory_error(exc) or index == len(attempts):
                raise
            print(f"Training failed with a memory-related error: {exc}")
            print("Retrying with more conservative settings...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if last_error is not None:
        raise last_error
    raise RuntimeError("Training did not start.")


def main() -> None:
    args = parse_args()
    results = run_training(args)

    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_model_path))
    metrics = best_model.val(data=args.data, imgsz=args.imgsz)
    metrics_dict = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "best_model_path": str(best_model_path),
        "results_dir": str(results.save_dir),
    }

    metrics_path = save_metrics(metrics_dict, Path(results.save_dir))
    print("\nTraining complete.")
    print(f"Results saved to: {results.save_dir}")
    print(f"Metrics saved to: {metrics_path}")
    print(json.dumps(metrics_dict, indent=2))


if __name__ == "__main__":
    main()
