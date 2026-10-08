"""
Segment a fixed set of classes in every image of a directory and save mask overlays.

Usage
-----
python examples/segment_directory.py IMAGE_DIR OUTPUT_DIR --backend florence2 \
    --weights florence-community/Florence-2-base --classes person bus

A class written as ``name:query,query`` (e.g. ``car:car,suv,taxi``) is queried by each phrase and reported as ``name``.
Box-prompted backends (``sam``, ``sam2``) also need ``--box-detector-backend`` and ``--box-detector-weights``.
"""

import argparse
from pathlib import Path

from image_container import ChannelOrder, ImageContainer
from PIL import Image

from open_vocabulary_segmentation import (
    DetectionThresholds,
    DetectorBackend,
    DetectorSettings,
    Device,
    MaskOverlayRenderer,
    OpenVocabularySegmenter,
    Prompt,
    PromptQuery,
    SegmentationResult,
    SegmentationThresholds,
    SegmenterBackend,
    SegmenterSettings,
    TextQuery,
    VisualQuery,
)

IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
CLASS_QUERY_SEPARATOR: str = ":"
QUERY_SEPARATOR: str = ","


def parse_prompt(class_arguments: list[str]) -> Prompt:
    """
    Build a text prompt from ``name`` or ``name:query,query`` arguments.

    A class given more than once collects the phrases of every argument, e.g. ``car:car,suv car:van``.

    Parameters
    ----------
    class_arguments : list[str]
        Command-line class arguments.

    Returns
    -------
    Prompt
        Classes named ``name``, each queried by its listed phrases or by its name.
    """
    class_texts: dict[str, list[str]] = {}
    for argument in class_arguments:
        class_name, _, phrases = argument.partition(CLASS_QUERY_SEPARATOR)
        stripped_name: str = class_name.strip()
        class_phrases: list[str] = phrases.split(QUERY_SEPARATOR) if phrases else [stripped_name]
        class_texts.setdefault(stripped_name, []).extend(class_phrases)
    return Prompt.from_texts(class_texts)


def describe_query(query: PromptQuery) -> str:
    """
    Short label of the query that matched an instance.

    Parameters
    ----------
    query : PromptQuery
        Matched query.

    Returns
    -------
    str
        The phrase of a text query, or the reference count of a visual query.
    """
    match query:
        case TextQuery(text=text):
            return text
        case VisualQuery(references=references):
            return f"<{len(references)} reference images>"


def build_box_detector(arguments: argparse.Namespace) -> DetectorSettings | None:
    """
    Settings of the box detector of a box-prompted backend.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed command-line arguments.

    Raises
    ------
    ValueError
        If the backend is box-prompted and no detector backend or weights are given.

    Returns
    -------
    DetectorSettings | None
        Box detector settings, or ``None`` for backends finding instances themselves.
    """
    backend: SegmenterBackend = arguments.backend
    if not backend.is_box_prompted:
        return None
    if arguments.box_detector_backend is None or arguments.box_detector_weights is None:
        raise ValueError(f"{backend} needs --box-detector-backend and --box-detector-weights.")
    return DetectorSettings(
        backend=arguments.box_detector_backend,
        weights_path=arguments.box_detector_weights,
        thresholds=DetectionThresholds(
            confidence_threshold=arguments.box_confidence_threshold, nms_iou_threshold=arguments.box_nms_iou_threshold
        ),
        batch_size=arguments.batch_size,
        device=arguments.device,
        is_half_precision_enabled=arguments.half,
    )


def main() -> None:
    """
    Parse arguments, run segmentation, print the instances of each image and save their overlays.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--backend", type=SegmenterBackend, choices=list(SegmenterBackend), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--nms-iou-threshold", type=float, default=None)
    parser.add_argument("--box-detector-backend", type=DetectorBackend, choices=list(DetectorBackend), default=None)
    parser.add_argument("--box-detector-weights", default=None)
    parser.add_argument("--box-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--box-nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=Device, choices=list(Device), default=Device.AUTO)
    parser.add_argument("--half", action="store_true")
    arguments: argparse.Namespace = parser.parse_args()

    image_directory: Path = arguments.image_directory
    image_paths: list[Path] = sorted(
        path for path in image_directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"no images found in {image_directory}")
    output_directory: Path = arguments.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    segmenter: OpenVocabularySegmenter = SegmenterSettings(
        backend=arguments.backend,
        weights_path=arguments.weights,
        thresholds=SegmentationThresholds(
            confidence_threshold=arguments.confidence_threshold, nms_iou_threshold=arguments.nms_iou_threshold
        ),
        box_detector=build_box_detector(arguments),
        batch_size=arguments.batch_size,
        device=arguments.device,
        is_half_precision_enabled=arguments.half,
    ).build()
    prompt: Prompt = parse_prompt(arguments.classes)

    images: list[Image.Image] = [Image.open(path) for path in image_paths]
    results: list[SegmentationResult] = segmenter.segment_images(images, prompt)
    renderer: MaskOverlayRenderer = MaskOverlayRenderer()

    for path, image, result in zip(image_paths, images, results, strict=True):
        print(f"{path.name}: {len(result)} instances")
        for segmentation in result:
            print(
                f"  {segmentation.class_name:20s} {describe_query(segmentation.matched_query):20s} "
                + f"{segmentation.confidence:.3f} {segmentation.area:8d}px {segmentation.box.value.round(1).tolist()}"
            )
        source = ImageContainer.register(image.convert("RGB"), ChannelOrder.RGB)
        renderer.render(source, result).save(str(output_directory / f"{path.stem}_overlay.png"))


if __name__ == "__main__":
    main()
