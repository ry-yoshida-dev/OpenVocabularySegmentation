import numpy as np
import pytest
from open_vocabulary_detector.backends.florence2 import LocationParser

from open_vocabulary_segmentation import ImageSize
from open_vocabulary_segmentation.array_types import BoolArray, FloatArray
from open_vocabulary_segmentation.backends.florence2 import LocationTokens, PolygonParser, PolygonRasterizer


def test_parse_reads_point_pairs_at_bin_centers() -> None:
    polygons: list[FloatArray] = PolygonParser.parse(
        "</s><s><loc_100><loc_200><loc_300><loc_200><loc_300><loc_400></s>"
    )
    assert len(polygons) == 1
    np.testing.assert_allclose(polygons[0], [[0.1005, 0.2005], [0.3005, 0.2005], [0.3005, 0.4005]])


def test_parse_splits_polygons_and_skips_degenerate_parts() -> None:
    polygons: list[FloatArray] = PolygonParser.parse(
        "</s><s><loc_1><loc_2><loc_3><loc_4><loc_5><loc_6><sep><loc_7><loc_8><loc_9><loc_10>"
        + "<poly><loc_10><loc_10><loc_20><loc_10><loc_20><loc_20><loc_99></poly></s>"
    )
    assert [polygon.shape for polygon in polygons] == [(3, 2), (3, 2)]
    assert polygons[1][2].tolist() == pytest.approx([0.0205, 0.0205])


def test_parse_without_polygons_is_empty() -> None:
    assert PolygonParser.parse("</s><s>cat</s>") == []


def test_rasterize_fills_the_union_of_polygons() -> None:
    image_size: ImageSize = ImageSize(width=20, height=10)
    square: FloatArray = np.array([[0.0, 0.0], [0.25, 0.0], [0.25, 0.5], [0.0, 0.5]])
    triangle: FloatArray = np.array([[0.75, 0.5], [1.0, 0.5], [1.0, 1.0]])
    mask: BoolArray = PolygonRasterizer.rasterize([square, triangle], image_size)
    assert mask.shape == (10, 20)
    assert mask[2, 2]
    assert mask[9, 19]
    assert not mask[8, 10]
    assert not PolygonRasterizer.rasterize([], image_size).any()


def test_location_tokens_round_trip_parsed_boxes() -> None:
    generated_text: str = "</s><s>cat<loc_8><loc_114><loc_497><loc_999></s>"
    normalized_xyxy: FloatArray = LocationParser.parse(generated_text)[0]
    assert LocationTokens.encode_box(normalized_xyxy) == "<loc_8><loc_114><loc_497><loc_999>"
    assert LocationTokens.encode_box(np.array([-0.1, 0.0, 1.0, 1.2])) == "<loc_0><loc_0><loc_999><loc_999>"
    with pytest.raises(ValueError, match="shape"):
        LocationTokens.encode_box(np.zeros(3))
