import dataclasses
import types
from enum import Enum
from pathlib import Path
from typing import cast, get_args, get_type_hints

import pytest
import yaml

import open_vocabulary_segmentation
from open_vocabulary_segmentation import (
    ConceptTrackerSettings,
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    Device,
    MaskGeneratorSettings,
    ObjectTrackerSettings,
    Precision,
    PromptableBackend,
    PromptableSegmenterSettings,
    SegmentationThresholds,
    SegmenterBackend,
    SegmenterSettings,
    TrackerBackend,
)

PRESET_DIRECTORY: Path = Path(open_vocabulary_segmentation.__file__).parent / "config"
SEGMENTER_PRESET_PATHS: list[Path] = sorted(PRESET_DIRECTORY.glob("*/*.yaml"))
FAMILY_SECTIONS: dict[str, tuple[str, type[object]]] = {
    "promptable": ("promptable_segmenter", PromptableSegmenterSettings),
    "tracker": ("object_tracker", ObjectTrackerSettings),
    "concept_tracker": ("concept_tracker", ConceptTrackerSettings),
    "generation": ("mask_generator", MaskGeneratorSettings),
}


def to_section(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    entries: dict[object, object] = cast(dict[object, object], value)
    return {str(key): item for key, item in entries.items()}


def load_section(path: Path, key: str) -> dict[str, object]:
    return to_section(to_section(yaml.safe_load(path.read_text(encoding="utf-8")))[key])


def dataclass_of(annotation: object) -> type[object] | None:
    candidates: tuple[object, ...] = get_args(annotation) if isinstance(annotation, types.UnionType) else (annotation,)
    for candidate in candidates:
        if isinstance(candidate, type) and dataclasses.is_dataclass(candidate):
            return candidate
    return None


def enum_of(annotation: object) -> type[Enum] | None:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    return None


def assert_matches_schema(section: dict[str, object], schema: type[object]) -> None:
    assert dataclasses.is_dataclass(schema)
    init_fields: list[dataclasses.Field[object]] = [field for field in dataclasses.fields(schema) if field.init]
    hints: dict[str, object] = get_type_hints(schema)
    assert set(section) <= {field.name for field in init_fields}
    required: set[str] = {
        field.name
        for field in init_fields
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    }
    assert required <= set(section)
    for name, value in section.items():
        nested: type[object] | None = dataclass_of(hints[name])
        enum_type: type[Enum] | None = enum_of(hints[name])
        if nested is not None and value is not None:
            assert_matches_schema(to_section(value), nested)
        elif enum_type is not None:
            assert value in {member.value for member in enum_type}


def test_every_segmenter_backend_has_presets() -> None:
    preset_backends: set[str] = {
        path.parent.name for path in SEGMENTER_PRESET_PATHS if path.parent.name not in FAMILY_SECTIONS
    }
    assert preset_backends == {backend.value for backend in SegmenterBackend}


@pytest.mark.parametrize(
    "path",
    [path for path in SEGMENTER_PRESET_PATHS if path.parent.name not in FAMILY_SECTIONS],
    ids=lambda path: f"{path.parent.name}/{path.stem}",
)
def test_segmenter_preset_matches_settings_schema(path: Path) -> None:
    section: dict[str, object] = load_section(path, "segmenter")
    backend: SegmenterBackend = SegmenterBackend(str(section["backend"]))
    assert backend.value == path.parent.name
    assert ("box_detector" in section) is backend.is_box_prompted
    assert_matches_schema(section, SegmenterSettings)
    assert section["precision"] in {precision.value for precision in Precision}
    if backend.is_box_prompted:
        box_detector: dict[str, object] = to_section(section["box_detector"])
        assert set(box_detector) == {field.name for field in dataclasses.fields(DetectorSettings)}
        assert box_detector["backend"] in {detector.value for detector in DetectorBackend}
        assert set(to_section(box_detector["thresholds"])) == {
            field.name for field in dataclasses.fields(DetectionThresholds)
        }


@pytest.mark.parametrize(
    "path",
    sorted(path for family in FAMILY_SECTIONS for path in (PRESET_DIRECTORY / family).glob("*/*.yaml")),
    ids=lambda path: f"{path.parent.parent.name}/{path.parent.name}/{path.stem}",
)
def test_family_preset_matches_settings_schema(path: Path) -> None:
    key, schema = FAMILY_SECTIONS[path.parent.parent.name]
    section: dict[str, object] = load_section(path, key)
    assert_matches_schema(section, schema)
    backend_section: dict[str, object] = (
        to_section(section["segmenter"]) if path.parent.parent.name == "generation" else section
    )
    if "backend" in backend_section:
        assert backend_section["backend"] == path.parent.name


def test_family_presets_cover_every_backend() -> None:
    def backends_of(family: str) -> set[str]:
        return {path.name for path in (PRESET_DIRECTORY / family).iterdir() if path.is_dir()}

    assert backends_of("promptable") == {backend.value for backend in PromptableBackend}
    assert backends_of("tracker") == {backend.value for backend in TrackerBackend}
    assert backends_of("concept_tracker") == {"sam3"}
    assert backends_of("generation") == {
        backend.value for backend in PromptableBackend if backend is not PromptableBackend.FLORENCE2
    }


def test_backend_capabilities() -> None:
    assert SegmenterBackend.SAM.is_box_prompted
    assert SegmenterBackend.SAM2.is_box_prompted
    assert not SegmenterBackend.SAM3.is_box_prompted
    assert not SegmenterBackend.FLORENCE2.is_box_prompted
    assert not SegmenterBackend.FLORENCE2_REFERRING.is_box_prompted
    assert SegmenterBackend.SAM.promptable_backend is PromptableBackend.SAM
    assert SegmenterBackend.SAM2.promptable_backend is PromptableBackend.SAM2
    assert SegmenterBackend.SAM3.promptable_backend is None
    assert not PromptableBackend.SAM.is_mask_only_prompt_supported
    assert PromptableBackend.SAM3.is_mask_only_prompt_supported
    assert "sam3_video" in TrackerBackend.SAM3.model_types


def test_box_detector_must_match_the_backend() -> None:
    thresholds: SegmentationThresholds = SegmentationThresholds(confidence_threshold=0.0, nms_iou_threshold=None)
    detector_settings: DetectorSettings = DetectorSettings(
        backend=DetectorBackend.GROUNDING_DINO,
        weights_path="IDEA-Research/grounding-dino-tiny",
        thresholds=DetectionThresholds(confidence_threshold=0.35, nms_iou_threshold=0.5),
    )
    with pytest.raises(ValueError, match="needs box_detector settings"):
        SegmenterSettings(backend=SegmenterBackend.SAM, weights_path="facebook/sam-vit-base", thresholds=thresholds)
    with pytest.raises(ValueError, match="takes no box_detector settings"):
        SegmenterSettings(
            backend=SegmenterBackend.SAM3,
            weights_path="facebook/sam3",
            thresholds=thresholds,
            box_detector=detector_settings,
        )
    settings: SegmenterSettings = SegmenterSettings(
        backend=SegmenterBackend.SAM2,
        weights_path="facebook/sam2.1-hiera-tiny",
        thresholds=thresholds,
        box_detector=detector_settings,
    )
    assert settings.box_detector == detector_settings


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
    with pytest.raises(ValueError, match="weights_path"):
        PromptableSegmenterSettings(backend=PromptableBackend.SAM, weights_path="")
    with pytest.raises(ValueError, match="weights_path"):
        ObjectTrackerSettings(backend=TrackerBackend.SAM2, weights_path=" ", device=Device.CPU)
    with pytest.raises(ValueError, match="confidence_threshold"):
        ConceptTrackerSettings(weights_path="facebook/sam3", confidence_threshold=2.0)
