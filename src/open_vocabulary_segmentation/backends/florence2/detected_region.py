from dataclasses import dataclass

from ...array_types import FloatArray


@dataclass(frozen=True, eq=False)
class DetectedRegion:
    """
    Box Florence-2 detected for one text query in one image of a mini-batch, waiting to be segmented.

    Attributes
    ----------
    image_index : int
        Index of the image in the mini-batch.
    query_id : int
        Index of the query in ``Prompt.queries``.
    normalized_xyxy : FloatArray
        XYXY box relative to the image size, shape (4,).

    Raises
    ------
    ValueError
        If the box does not have four coordinates.
    """

    image_index: int
    query_id: int
    normalized_xyxy: FloatArray

    def __post_init__(self) -> None:
        if self.normalized_xyxy.shape != (4,):
            raise ValueError(f"normalized_xyxy must have shape (4,). got {self.normalized_xyxy.shape}")
