from pathlib import Path
import time

import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V3_Large_Weights,
    EfficientNet_B0_Weights
)

from data_loader import test_loader, DEVICE


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

NUM_CLASSES = 13

MODEL_PATHS = {
    "ResNet50": MODEL_DIR / "resnet50_best.pth",
    "MobileNetV3": MODEL_DIR / "mobilenetv3_best.pth",
    "EfficientNet-B0": MODEL_DIR / "efficientnet_b0_best.pth"
}


# --------------------------------------------------
# 3. Create model
# --------------------------------------------------

def create_model(model_name):

    if model_name == "ResNet50":

        weights = ResNet50_Weights.DEFAULT

        model = models.resnet50(
            weights=weights
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif model_name == "MobileNetV3":

        weights = MobileNet_V3_Large_Weights.DEFAULT

        model = models.mobilenet_v3_large(
            weights=weights
        )

        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            NUM_CLASSES
        )

    elif model_name == "EfficientNet-B0":

        weights = EfficientNet_B0_Weights.DEFAULT

        model = models.efficientnet_b0(
            weights=weights
        )

        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            NUM_CLASSES
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# --------------------------------------------------
# 4. Parameter count
# --------------------------------------------------

def count_parameters(model):

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# --------------------------------------------------
# 5. Measure inference time
# --------------------------------------------------

def measure_inference_time(model):

    model.eval()

    # Get one batch
    images, _ = next(iter(test_loader))

    images = images.to(DEVICE)

    # Warm-up
    with torch.no_grad():

        for _ in range(5):

            _ = model(images)

    # Synchronize MPS before timing
    if DEVICE.type == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():

        for _ in range(20):

            _ = model(images)

    # Synchronize MPS after timing
    if DEVICE.type == "mps":
        torch.mps.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time

    total_images = images.size(0) * 20

    time_per_image = (
        total_time / total_images
    )

    images_per_second = (
        total_images / total_time
    )

    return time_per_image, images_per_second


# --------------------------------------------------
# 6. Main comparison
# --------------------------------------------------

print("\nModel Efficiency Comparison")
print("=" * 70)

print("Device:", DEVICE)

results = []


for model_name, model_path in MODEL_PATHS.items():

    print(
        f"\nLoading {model_name}..."
    )

    # Load architecture
    model = create_model(
        model_name
    )

    # Load trained checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    # ------------------------------
    # Parameters
    # ------------------------------

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    # ------------------------------
    # Model size
    # ------------------------------

    model_size_mb = (
        model_path.stat().st_size
        / (1024 * 1024)
    )

    # ------------------------------
    # Inference
    # ------------------------------

    time_per_image, images_per_second = (
        measure_inference_time(model)
    )

    # ------------------------------
    # Validation accuracy
    # ------------------------------

    validation_accuracy = (
        checkpoint["validation_accuracy"]
    )

    # ------------------------------
    # Print
    # ------------------------------

    print(
        f"Parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable Parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Model Size: "
        f"{model_size_mb:.2f} MB"
    )

    print(
        f"Best Validation Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"Inference Time/Image: "
        f"{time_per_image * 1000:.3f} ms"
    )

    print(
        f"Inference Speed: "
        f"{images_per_second:.2f} images/sec"
    )

    results.append(
        {
            "model": model_name,
            "parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "size_mb": model_size_mb,
            "validation_accuracy": validation_accuracy,
            "time_per_image_ms": time_per_image * 1000,
            "images_per_second": images_per_second
        }
    )

    # Free model memory
    del model

    if DEVICE.type == "mps":
        torch.mps.empty_cache()


# --------------------------------------------------
# 7. Final comparison table
# --------------------------------------------------

print("\n\nFINAL MODEL COMPARISON")
print("=" * 110)

print(
    f"{'Model':<18}"
    f"{'Parameters':>15}"
    f"{'Size (MB)':>12}"
    f"{'Val Acc':>12}"
    f"{'ms/image':>14}"
    f"{'Images/sec':>15}"
)

print("-" * 110)


for result in results:

    print(
        f"{result['model']:<18}"
        f"{result['parameters']:>15,}"
        f"{result['size_mb']:>12.2f}"
        f"{result['validation_accuracy'] * 100:>11.2f}%"
        f"{result['time_per_image_ms']:>14.3f}"
        f"{result['images_per_second']:>15.2f}"
    )


print("=" * 110)


# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "model_efficiency_comparison.txt"
)


with open(
    RESULTS_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AgriGuard AI - Model Efficiency Comparison\n"
    )

    file.write(
        "=" * 110 + "\n\n"
    )

    file.write(
        f"Device: {DEVICE}\n\n"
    )

    file.write(
        f"{'Model':<18}"
        f"{'Parameters':>15}"
        f"{'Size (MB)':>12}"
        f"{'Val Acc':>12}"
        f"{'ms/image':>14}"
        f"{'Images/sec':>15}\n"
    )

    file.write(
        "-" * 110 + "\n"
    )

    for result in results:

        file.write(
            f"{result['model']:<18}"
            f"{result['parameters']:>15,}"
            f"{result['size_mb']:>12.2f}"
            f"{result['validation_accuracy'] * 100:>11.2f}%"
            f"{result['time_per_image_ms']:>14.3f}"
            f"{result['images_per_second']:>15.2f}\n"
        )

    file.write(
        "=" * 110 + "\n"
    )


print(
    f"\nResults saved to:\n"
    f"{RESULTS_PATH}"
)