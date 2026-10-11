"""
Segment everything in an image (automatic mask generation) and save the regions as an overlay.

Usage
-----
python examples/generate_masks.py IMAGE OUTPUT --backend sam2 --weights facebook/sam2.1-hiera-tiny \
    --points-per-side 32 --crop-layers 1 --minimum-region-area 100
"""

import argparse
from pathlib import Path

from image_container import ChannelOrder, ImageContainer
from PIL import Image, ImageOps

from open_vocabulary_segmentation import (
    AutomaticMaskGenerator,
    Device,
    MaskGenerationOptions,
    MaskGenerationResult,
    MaskGeneratorSettings,
    MaskOverlayRenderer,
    Precision,
    PromptableBackend,
    PromptableSegmenterSettings,
    PromptElement,
)


def main() -> None:
    """
    Parse arguments, generate the masks, print them and save the overlay.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    point_backends: list[PromptableBackend] = [
        backend for backend in PromptableBackend if PromptElement.POINT in backend.supported_prompt_elements
    ]
    parser.add_argument("--backend", type=PromptableBackend, choices=point_backends, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--predicted-iou-threshold", type=float, default=0.88)
    parser.add_argument("--stability-score-threshold", type=float, default=0.95)
    parser.add_argument("--crop-layers", type=int, default=0)
    parser.add_argument("--minimum-region-area", type=int, default=0)
    parser.add_argument("--device", type=Device, choices=list(Device), default=Device.AUTO)
    parser.add_argument("--precision", type=Precision, choices=list(Precision), default=Precision.FLOAT32)
    arguments: argparse.Namespace = parser.parse_args()

    generator: AutomaticMaskGenerator = MaskGeneratorSettings(
        segmenter=PromptableSegmenterSettings(
            backend=arguments.backend,
            weights_path=arguments.weights,
            device=arguments.device,
            precision=arguments.precision,
        ),
        options=MaskGenerationOptions(
            points_per_side=arguments.points_per_side,
            points_per_batch=arguments.points_per_batch,
            predicted_iou_threshold=arguments.predicted_iou_threshold,
            stability_score_threshold=arguments.stability_score_threshold,
            crop_layer_count=arguments.crop_layers,
            minimum_region_area=arguments.minimum_region_area,
        ),
    ).build()
    image: Image.Image = ImageOps.exif_transpose(Image.open(arguments.image)).convert("RGB")
    result: MaskGenerationResult = generator.generate(image).sort_by_area()
    print(f"{arguments.image.name}: {len(result)} regions")
    for region in result:
        print(f"  {region.area:8d}px score {region.score:.3f} stability {region.stability_score:.3f}")
    source = ImageContainer.register(image, ChannelOrder.RGB)
    MaskOverlayRenderer().render_generated(source, result).save(str(arguments.output))


if __name__ == "__main__":
    main()
