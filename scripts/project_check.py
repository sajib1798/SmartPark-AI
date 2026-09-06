from pathlib import Path

import torch

from app.config import settings


# ============================================================
# DISPLAY RESULT
# ============================================================

def show_result(
    name: str,
    passed: bool,
    details: str = "",
) -> None:
    """
    Print a project health-check result.
    """

    status = "PASS" if passed else "FAIL"

    print(
        f"[{status}] {name}"
    )

    if details:

        print(
            f"       {details}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Perform final SmartPark AI project checks.
    """

    print("=" * 70)

    print(
        "SmartPark AI - Final Project Check"
    )

    print("=" * 70)

    print()

    results = []

    # --------------------------------------------------------
    # Project directories
    # --------------------------------------------------------

    directory_checks = {
        "App directory":
            settings.APP_DIR,

        "Data directory":
            settings.DATA_DIR,

        "Models directory":
            settings.MODELS_DIR,

        "Outputs directory":
            settings.OUTPUT_DIR,

        "Config directory":
            settings.CONFIG_DIR,
    }

    for name, path in (
        directory_checks.items()
    ):

        passed = path.exists()

        results.append(
            passed
        )

        show_result(
            name=name,
            passed=passed,
            details=str(path),
        )

    print()

    # --------------------------------------------------------
    # Dataset YAML
    # --------------------------------------------------------

    dataset_yaml_exists = (
        settings.DATA_YAML_FILE.exists()
    )

    results.append(
        dataset_yaml_exists
    )

    show_result(
        name="Dataset configuration",
        passed=dataset_yaml_exists,
        details=str(
            settings.DATA_YAML_FILE
        ),
    )

    # --------------------------------------------------------
    # Custom model
    # --------------------------------------------------------

    model_exists = (
        settings.BEST_MODEL_FILE.exists()
    )

    results.append(
        model_exists
    )

    show_result(
        name="Trained best.pt",
        passed=model_exists,
        details=str(
            settings.BEST_MODEL_FILE
        ),
    )

    # --------------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------------

    metrics_exist = (
        settings
        .EVALUATION_METRICS_FILE
        .exists()
    )

    results.append(
        metrics_exist
    )

    show_result(
        name="Evaluation metrics",
        passed=metrics_exist,
        details=str(
            settings
            .EVALUATION_METRICS_FILE
        ),
    )

    print()

    # --------------------------------------------------------
    # PyTorch
    # --------------------------------------------------------

    show_result(
        name="PyTorch installed",
        passed=True,
        details=torch.__version__,
    )

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    cuda_available = (
        torch.cuda.is_available()
    )

    show_result(
        name="CUDA available",
        passed=cuda_available,
        details=(
            torch.cuda.get_device_name(0)
            if cuda_available
            else "CPU only"
        ),
    )

    print()

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    critical_passed = all(
        results
    )

    print("=" * 70)

    if critical_passed:

        print(
            "SmartPark AI is ready "
            "for final CDC presentation."
        )

    else:

        print(
            "SmartPark AI still has "
            "missing project components."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()