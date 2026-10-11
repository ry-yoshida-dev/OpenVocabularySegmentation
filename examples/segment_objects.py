"""
Segment objects marked by points and boxes in one image and save the masks with their prompts drawn.

Usage
-----
python examples/segment_objects.py IMAGE OUTPUT --backend sam2 --weights facebook/sam2.1-hiera-tiny \
    --object "point=420,310 point=380,290 negative=600,300" --object "box=50,60,300,400"

Each ``--object`` is one object prompt: any number of ``point=x,y`` (on the object), ``negative=x,y`` (off the
object) and at most one ``box=x_min,y_min,x_max,y_max``. ``--refine "negative=x,y"`` refines the first object with
corrective clicks and its previous mask logits, the way an interactive tool does.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from geometry import Points2D
from image_container import ChannelOrder, ImageContainer
from PIL import Image, ImageOps

from open_vocabulary_segmentation import (
    Device,
    EncodedImage,
    MaskOutput,
    MaskOverlayRenderer,
    ObjectPrompt,
    ObjectSegmentationResult,
    Precision,
    PromptableBackend,
    PromptableSegmenter,
    PromptableSegmenterSettings,
)

ELEMENT_SEPARATOR: str = "="
COORDINATE_SEPARATOR: str = ","


@dataclass
class PromptElements:
    """
    Points and box read from one ``--object`` or ``--refine`` argument.

    Attributes
    ----------
    positive_points : list[tuple[float, float]]
        Points on the object.
    negative_points : list[tuple[float, float]]
        Points off the object.
    box_xyxy : tuple[float, float, float, float] | None
        Box around the object, if given.
    """

    positive_points: list[tuple[float, float]] = field(default_factory=list)
    negative_points: list[tuple[float, float]] = field(default_factory=list)
    box_xyxy: tuple[float, float, float, float] | None = None

    @staticmethod
    def points_of(coordinates: list[tuple[float, float]]) -> Points2D | None:
        """
        Points as ``geometry.Points2D``.

        Parameters
        ----------
        coordinates : list[tuple[float, float]]
            ``(x, y)`` points.

        Returns
        -------
        Points2D | None
            Points, or ``None`` without any.
        """
        return Points2D(value=np.array(coordinates, dtype=np.float64)) if coordinates else None


def parse_coordinates(text: str, count: int) -> tuple[float, ...]:
    """
    Read comma-separated numbers.

    Parameters
    ----------
    text : str
        Numbers such as ``"420,310"``.
    count : int
        Expected number of values.

    Raises
    ------
    ValueError
        If the number of values differs from ``count``.

    Returns
    -------
    tuple[float, ...]
        Parsed values.
    """
    values: tuple[float, ...] = tuple(float(value) for value in text.split(COORDINATE_SEPARATOR))
    if len(values) != count:
        raise ValueError(f"expected {count} numbers. got {text!r}")
    return values


def parse_elements(description: str) -> PromptElements:
    """
    Read ``point=x,y negative=x,y box=x_min,y_min,x_max,y_max`` elements.

    Parameters
    ----------
    description : str
        Space-separated prompt elements.

    Raises
    ------
    ValueError
        If an element is unknown or malformed.

    Returns
    -------
    PromptElements
        Parsed points and box.
    """
    elements: PromptElements = PromptElements()
    for element in description.split():
        key, _, value = element.partition(ELEMENT_SEPARATOR)
        match key:
            case "point":
                x, y = parse_coordinates(value, 2)
                elements.positive_points.append((x, y))
            case "negative":
                x, y = parse_coordinates(value, 2)
                elements.negative_points.append((x, y))
            case "box":
                x_min, y_min, x_max, y_max = parse_coordinates(value, 4)
                elements.box_xyxy = (x_min, y_min, x_max, y_max)
            case _:
                raise ValueError(f"unknown prompt element {element!r}; use point=, negative= or box=.")
    return elements


def parse_object(description: str) -> ObjectPrompt:
    """
    Build an object prompt from one ``--object`` argument.

    Parameters
    ----------
    description : str
        Space-separated prompt elements.

    Returns
    -------
    ObjectPrompt
        Prompt with the described points and box.
    """
    elements: PromptElements = parse_elements(description)
    return ObjectPrompt.from_coordinates(
        positive_points=elements.positive_points,
        negative_points=elements.negative_points,
        box_xyxy=elements.box_xyxy,
    )


def print_result(title: str, result: ObjectSegmentationResult) -> None:
    """
    Print every mask of a result.

    Parameters
    ----------
    title : str
        Heading of the listing.
    result : ObjectSegmentationResult
        Masks to print.
    """
    print(title)
    for segmentation in result:
        box: str = "empty" if segmentation.box is None else str(segmentation.box.value.round(1).tolist())
        print(
            f"  prompt {segmentation.prompt_index}: score {segmentation.score:.3f} "
            + f"stability {segmentation.stability_score:.3f} {segmentation.area:8d}px {box}"
        )


def main() -> None:
    """
    Parse arguments, segment the prompted objects, optionally refine the first one, and save the overlay.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", type=PromptableBackend, choices=list(PromptableBackend), required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--object", dest="objects", action="append", required=True)
    parser.add_argument("--refine", default=None)
    parser.add_argument("--mask-output", type=MaskOutput, choices=list(MaskOutput), default=MaskOutput.AUTO)
    parser.add_argument("--device", type=Device, choices=list(Device), default=Device.AUTO)
    parser.add_argument("--precision", type=Precision, choices=list(Precision), default=Precision.FLOAT32)
    arguments: argparse.Namespace = parser.parse_args()

    segmenter: PromptableSegmenter = PromptableSegmenterSettings(
        backend=arguments.backend,
        weights_path=arguments.weights,
        device=arguments.device,
        precision=arguments.precision,
    ).build()
    image: Image.Image = ImageOps.exif_transpose(Image.open(arguments.image)).convert("RGB")
    encoding: EncodedImage = segmenter.encode(image)
    prompts: list[ObjectPrompt] = [parse_object(description) for description in arguments.objects]
    result: ObjectSegmentationResult = segmenter.segment(encoding, prompts, arguments.mask_output)
    print_result(f"{arguments.image.name}: {len(result)} masks", result)

    if arguments.refine is not None:
        correction: PromptElements = parse_elements(arguments.refine)
        refined_prompt: ObjectPrompt = (
            result.of_prompt(0)
            .best_per_prompt()[0]
            .refine(
                PromptElements.points_of(correction.positive_points),
                PromptElements.points_of(correction.negative_points),
            )
        )
        result = segmenter.segment(encoding, [refined_prompt, *prompts[1:]], arguments.mask_output)
        print_result("after refinement", result)

    source = ImageContainer.register(image, ChannelOrder.RGB)
    MaskOverlayRenderer().render_objects(source, result).save(str(arguments.output))


if __name__ == "__main__":
    main()
