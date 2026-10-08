from dataclasses import fields
from pathlib import Path
from typing import cast

import pytest
import yaml

import open_vocabulary_segmentation
from open_vocabulary_segmentation import (
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    Device,
    SegmentationThresholds,
    SegmenterBackend,
    SegmenterSettings,
)

PRESET_DIRECTORY: Path = Path(open_vocabulary_segmentation.__file__).parent / "config"
PRESET_PATHS: list[Path] = sorted(PRESET_DIRECTORY.glob("*/*.yaml"))


def to_section(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    entries: dict[object, object] = cast(dict[object, object], value)
    return {str(key): item for key, item in entries.items()}


def load_preset_section(path: Path) -> dict[str, object]:
    document: dict[str, object] = to_section(yaml.safe_load(path.read_text(encoding="utf-8")))
    return to_section(document["segmenter"])


def build_detector_settings() -> DetectorSettings:
    return DetectorSettings(
        backend=DetectorBackend.GROUNDING_DINO,
        weights_path="IDEA-Research/grounding-dino-tiny",
        thresholds=DetectionThresholds(confidence_threshold=0.35, nms_iou_threshold=0.5),
    )


def test_every_backend_has_presets() -> None:
    preset_backends: set[str] = {path.parent.name for path in PRESET_PATHS}
    assert preset_backends == {backend.value for backend in SegmenterBackend}


@pytest.mark.parametrize("path", PRESET_PATHS, ids=lambda path: f"{path.parent.name}/{path.stem}")
def test_preset_matches_settings_schema(path: Path) -> None:
    section: dict[str, object] = load_preset_section(path)
    backend: SegmenterBackend = SegmenterBackend(str(section["backend"]))
    expected_fields: set[str] = {field.name for field in fields(SegmenterSettings)}
    if not backend.is_box_prompted:
        expected_fields.discard("box_detector")
    assert set(section) == expected_fields
    assert backend.value == path.parent.name
    assert section["device"] in {device.value for device in Device}
    assert set(to_section(section["thresholds"])) == {field.name for field in fields(SegmentationThresholds)}


@pytest.mark.parametrize(
    "path",
    [path for path in PRESET_PATHS if SegmenterBackend(path.parent.name).is_box_prompted],
    ids=lambda path: f"{path.parent.name}/{path.stem}",
)
def test_box_detector_section_matches_detector_settings_schema(path: Path) -> None:
    box_detector: dict[str, object] = to_section(load_preset_section(path)["box_detector"])
    assert set(box_detector) == {field.name for field in fields(DetectorSettings)}
    assert box_detector["backend"] in {backend.value for backend in DetectorBackend}
    assert set(to_section(box_detector["thresholds"])) == {field.name for field in fields(DetectionThresholds)}


def test_backend_capabilities() -> None:
    assert SegmenterBackend.SAM.is_box_prompted
    assert SegmenterBackend.SAM2.is_box_prompted
    assert not SegmenterBackend.SAM3.is_box_prompted
    assert not SegmenterBackend.FLORENCE2.is_box_prompted


def test_box_detector_must_match_the_backend() -> None:
    thresholds: SegmentationThresholds = SegmentationThresholds(confidence_threshold=0.0, nms_iou_threshold=None)
    with pytest.raises(ValueError, match="needs box_detector settings"):
        SegmenterSettings(backend=SegmenterBackend.SAM, weights_path="facebook/sam-vit-base", thresholds=thresholds)
    with pytest.raises(ValueError, match="takes no box_detector settings"):
        SegmenterSettings(
            backend=SegmenterBackend.SAM3,
            weights_path="facebook/sam3",
            thresholds=thresholds,
            box_detector=build_detector_settings(),
        )
    settings: SegmenterSettings = SegmenterSettings(
        backend=SegmenterBackend.SAM2,
        weights_path="facebook/sam2.1-hiera-tiny",
        thresholds=thresholds,
        box_detector=build_detector_settings(),
    )
    assert settings.box_detector == build_detector_settings()


def test_invalid_settings_raise() -> None:
    thresholds: SegmentationThresholds = SegmentationThresholds(confidence_threshold=0.5, nms_iou_threshold=0.5)
    with pytest.raises(ValueError, match="batch_size"):
        SegmenterSettings(
            backend=SegmenterBackend.SAM3, weights_path="facebook/sam3", thresholds=thresholds, batch_size=0
        )
    with pytest.raises(ValueError, match="weights_path"):
        SegmenterSettings(backend=SegmenterBackend.SAM3, weights_path=" ", thresholds=thresholds)
    with pytest.raises(ValueError, match="confidence_threshold"):
        SegmentationThresholds(confidence_threshold=1.5, nms_iou_threshold=None)
    with pytest.raises(ValueError, match="nms_iou_threshold"):
        SegmentationThresholds(confidence_threshold=0.5, nms_iou_threshold=-0.1)
