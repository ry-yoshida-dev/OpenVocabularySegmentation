import subprocess
import sys

BACKEND_MODULES: tuple[str, ...] = (
    "transformers",
    "open_vocabulary_detector.backends.grounding_dino",
    "open_vocabulary_segmentation.backends.sam",
    "open_vocabulary_segmentation.backends.sam2",
    "open_vocabulary_segmentation.backends.sam3",
    "open_vocabulary_segmentation.backends.florence2",
)


def test_package_import_loads_no_backend() -> None:
    script: str = (
        "import sys\n"
        + "import open_vocabulary_segmentation\n"
        + f"print(','.join(name for name in {BACKEND_MODULES!r} if name in sys.modules))\n"
    )
    completed: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == ""
