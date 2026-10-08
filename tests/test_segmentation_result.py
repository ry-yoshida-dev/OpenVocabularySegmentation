import numpy as np
import pytest
from geometry import Box2DFormat
from image_container import BinaryImage

from open_vocabulary_segmentation import ImageSize, Prompt, SegmentationResult, TextQuery
from open_vocabulary_segmentation.array_types import BoolArray, IntArray
from open_vocabulary_segmentation.result import MaskOverlap

IMAGE_SIZE: ImageSize = ImageSize(width=10, height=8)
PROMPT: Prompt = Prompt.from_texts({"car": ("car", "suv"), "dog": ("dog",)})


def rectangle(x_min: int, y_min: int, x_max: int, y_max: int) -> BoolArray:
    mask: BoolArray = np.zeros((IMAGE_SIZE.height, IMAGE_SIZE.width), dtype=np.bool_)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def build_result() -> SegmentationResult:
    return SegmentationResult.from_masks(
        masks=np.stack(
            [
                rectangle(0, 0, 4, 4),
                rectangle(0, 0, 4, 5),
                np.zeros((IMAGE_SIZE.height, IMAGE_SIZE.width), dtype=np.bool_),
                rectangle(1, 1, 4, 4),
                rectangle(6, 2, 10, 8),
            ]
        ),
        confidences=np.array([0.6, 0.9, 0.95, 0.7, 0.3], dtype=np.float64),
        query_ids=np.array([0, 1, 0, 2, 2], dtype=np.int64),
        prompt=PROMPT,
        image_size=IMAGE_SIZE,
    )


def test_from_masks_drops_empty_masks_and_derives_boxes() -> None:
    result: SegmentationResult = build_result()
    assert len(result) == 4
    assert result.confidences.tolist() == pytest.approx([0.6, 0.9, 0.7, 0.3])
    assert result.xyxy.tolist() == [[0, 0, 4, 4], [0, 0, 4, 5], [1, 1, 4, 4], [6, 2, 10, 8]]
    assert result.boxes.box_format is Box2DFormat.XYXY
    assert result.areas.tolist() == [16, 20, 9, 24]
    assert result.class_ids.tolist() == [0, 0, 1, 1]


def test_filter_and_sort() -> None:
    result: SegmentationResult = build_result()
    assert result.filter_by_confidence(0.65).confidences.tolist() == pytest.approx([0.9, 0.7])
    assert result.sort_by_confidence().confidences.tolist() == pytest.approx([0.9, 0.7, 0.6, 0.3])


def test_class_wise_non_maximum_suppression_merges_queries_of_one_class() -> None:
    merged: SegmentationResult = build_result().non_maximum_suppression(0.5)
    assert merged.confidences.tolist() == pytest.approx([0.9, 0.7, 0.3])
    assert [segmentation.matched_query for segmentation in merged] == [
        TextQuery("suv"),
        TextQuery("dog"),
        TextQuery("dog"),
    ]


def test_class_agnostic_non_maximum_suppression() -> None:
    merged: SegmentationResult = build_result().non_maximum_suppression(0.4, is_class_agnostic=True)
    assert merged.confidences.tolist() == pytest.approx([0.9, 0.3])


def test_iteration_yields_masks_and_geometry_boxes() -> None:
    segmentations = list(build_result())
    assert segmentations[0].class_name == "car"
    assert segmentations[0].area == 16
    assert float(segmentations[0].box.area) == pytest.approx(16.0)
    assert segmentations[0].mask.shape == (IMAGE_SIZE.height, IMAGE_SIZE.width)


def test_class_map_lets_the_most_confident_instance_win() -> None:
    class_map: IntArray = build_result().class_map()
    assert class_map.shape == (IMAGE_SIZE.height, IMAGE_SIZE.width)
    assert class_map[2, 2] == 0
    assert class_map[4, 0] == 0
    assert class_map[5, 7] == 1
    assert class_map[7, 0] == -1


def test_empty_and_validation() -> None:
    empty: SegmentationResult = SegmentationResult.empty(PROMPT, IMAGE_SIZE)
    assert len(empty) == 0
    assert empty.xyxy.shape == (0, 4)
    assert empty.areas.shape == (0,)
    assert empty.non_maximum_suppression(0.5) is empty
    assert empty.class_map().tolist() == np.full((8, 10), -1).tolist()
    with pytest.raises(ValueError, match="must not be empty"):
        SegmentationResult(
            masks=np.zeros((1, 8, 10), dtype=np.bool_),
            confidences=np.array([0.5]),
            query_ids=np.array([0]),
            prompt=PROMPT,
            image_size=IMAGE_SIZE,
        )
    with pytest.raises(ValueError, match="masks must have shape"):
        SegmentationResult(
            masks=np.zeros((0, 3, 3), dtype=np.bool_),
            confidences=np.zeros(0),
            query_ids=np.zeros(0, dtype=np.int64),
            prompt=PROMPT,
            image_size=IMAGE_SIZE,
        )
    with pytest.raises(ValueError, match="query_ids must be in"):
        SegmentationResult.from_masks(
            masks=rectangle(0, 0, 2, 2)[None],
            confidences=np.array([0.5]),
            query_ids=np.array([3]),
            prompt=PROMPT,
            image_size=IMAGE_SIZE,
        )


def test_mask_overlap_counts_only_inside_box_intersections() -> None:
    masks: BoolArray = np.stack([rectangle(0, 0, 4, 4), rectangle(2, 2, 6, 6), rectangle(7, 6, 9, 8)])
    pixel_xyxy: IntArray = np.array([[0, 0, 4, 4], [2, 2, 6, 6], [7, 6, 9, 8]], dtype=np.int64)
    overlaps = MaskOverlap.pairwise_iou(masks, pixel_xyxy)
    assert overlaps[0, 1] == pytest.approx(4 / 28)
    assert overlaps[1, 0] == pytest.approx(4 / 28)
    assert overlaps[0, 2] == 0.0
    assert np.diag(overlaps).tolist() == [1.0, 1.0, 1.0]


def test_top_k_keeps_the_most_confident_instances() -> None:
    result: SegmentationResult = build_result()
    assert result.top_k(2).confidences.tolist() == pytest.approx([0.9, 0.7])
    assert len(result.top_k(10)) == 4
    assert len(result.top_k(0)) == 0
    with pytest.raises(ValueError, match="count"):
        result.top_k(-1)


def test_filter_by_class_and_query() -> None:
    result: SegmentationResult = build_result()
    assert result.filter_by_class("dog").confidences.tolist() == pytest.approx([0.7, 0.3])
    assert result.filter_by_class(" car ").confidences.tolist() == pytest.approx([0.6, 0.9])
    assert len(result.filter_by_classes(("car", "dog"))) == 4
    assert result.filter_by_query(TextQuery("suv")).confidences.tolist() == pytest.approx([0.9])
    with pytest.raises(KeyError, match="'cat' is not a class"):
        result.filter_by_class("cat")
    with pytest.raises(KeyError, match="is not a query"):
        result.filter_by_query(TextQuery("taxi"))


def test_filter_by_area() -> None:
    result: SegmentationResult = build_result()
    assert result.filter_by_area(16).areas.tolist() == [16, 20, 24]
    assert result.filter_by_area(10, 20).areas.tolist() == [16, 20]
    with pytest.raises(ValueError, match="maximum_area"):
        result.filter_by_area(10, 5)
    with pytest.raises(ValueError, match="minimum_area"):
        result.filter_by_area(-1)


def test_class_masks_union_instances_of_each_class() -> None:
    result: SegmentationResult = build_result()
    car_mask: BinaryImage = result.class_mask("car")
    assert car_mask.shape == (IMAGE_SIZE.height, IMAGE_SIZE.width)
    assert car_mask.sum == 20
    assert result.class_mask("dog").sum == 9 + 24
    assert {name: mask.sum for name, mask in result.class_masks().items()} == {"car": 20, "dog": 33}
    assert result.union_mask().sum == 20 + 24
    assert SegmentationResult.empty(PROMPT, IMAGE_SIZE).class_mask("car").sum == 0


def test_instance_map_and_class_counts() -> None:
    result: SegmentationResult = build_result()
    instance_map: IntArray = result.instance_map()
    assert instance_map[2, 2] == 1
    assert instance_map[5, 7] == 3
    assert instance_map[7, 0] == -1
    assert result.class_counts() == {"car": 2, "dog": 2}
    assert result.filter_by_class("dog").class_counts() == {"car": 0, "dog": 2}
