import numpy as np
import pytest
from geometry import Box2DFormat, Boxes2D
from PIL import Image

from open_vocabulary_segmentation import Prompt, TextQuery, VisualQuery, VisualReference

RUN_SLOW_OPTION: str = "--run-slow"
SLOW_MARKER: str = "slow"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(RUN_SLOW_OPTION, action="store_true", default=False, help="run tests that load real models")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption(RUN_SLOW_OPTION):
        return
    skip_slow: pytest.MarkDecorator = pytest.mark.skip(reason=f"needs {RUN_SLOW_OPTION}")
    for item in items:
        if SLOW_MARKER in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def mixed_prompt() -> Prompt:
    reference: VisualReference = VisualReference(
        image=Image.new("RGB", (8, 8)),
        boxes=Boxes2D.register(value=np.array([[0.0, 0.0, 4.0, 4.0]]), box2d_format=Box2DFormat.XYXY),
    )
    return Prompt({"car": (TextQuery("car"), VisualQuery((reference,))), "dog": (TextQuery("dog"),)})
