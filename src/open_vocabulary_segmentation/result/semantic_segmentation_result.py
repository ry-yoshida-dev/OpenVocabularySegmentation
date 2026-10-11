from dataclasses import dataclass

import numpy as np
from image_container import BinaryImage
from open_vocabulary_detector import ImageSize, Prompt

from ..array_types import BoolArray, Float32Array, IntArray


@dataclass(frozen=True, eq=False)
class SemanticSegmentationResult:
    """
    Per-pixel probability of every class of a prompt, from a semantic segmentation head (SAM 3).

    Unlike instance masks, the semantic head scores every pixel for the concept as a whole, so touching instances
    are not separated; the probability of a class is the highest probability of its queries.

    Attributes
    ----------
    class_probabilities : Float32Array
        Probability of each class at each pixel in ``[0, 1]``, shape (C, H, W) in class-id order.
    prompt : Prompt
        Prompt whose classes were segmented.
    image_size : ImageSize
        Size of the source image.

    Raises
    ------
    ValueError
        If the probabilities do not match the classes and the image size or lie outside ``[0, 1]``.
    """

    class_probabilities: Float32Array
    prompt: Prompt
    image_size: ImageSize

    def __post_init__(self) -> None:
        expected_shape: tuple[int, int, int] = (
            len(self.prompt.class_names),
            self.image_size.height,
            self.image_size.width,
        )
        if self.class_probabilities.shape != expected_shape:
            raise ValueError(
                f"class_probabilities must have shape {expected_shape}. got {self.class_probabilities.shape}"
            )
        if bool(((self.class_probabilities < 0.0) | (self.class_probabilities > 1.0)).any()):
            raise ValueError("class_probabilities must be in [0, 1].")

    @property
    def class_names(self) -> tuple[str, ...]:
        """
        Output class names of the prompt.

        Returns
        -------
        tuple[str, ...]
            Names in class-id order.
        """
        return self.prompt.class_names

    def probability_of(self, class_name: str) -> Float32Array:
        """
        Probability map of one class.

        Parameters
        ----------
        class_name : str
            Output class name of the prompt.

        Raises
        ------
        KeyError
            If the prompt has no class ``class_name``.

        Returns
        -------
        Float32Array
            Probability at each pixel, shape (H, W).
        """
        stripped_name: str = class_name.strip()
        if stripped_name not in self.class_names:
            raise KeyError(f"{class_name!r} is not a class of the prompt. classes: {list(self.class_names)}")
        probabilities: Float32Array = self.class_probabilities[self.class_names.index(stripped_name)]
        return probabilities

    def class_mask(self, class_name: str, threshold: float = 0.5) -> BinaryImage:
        """
        Pixels of one class.

        Parameters
        ----------
        class_name : str
            Output class name of the prompt.
        threshold : float, optional
            Minimum probability of a class pixel.

        Raises
        ------
        KeyError
            If the prompt has no class ``class_name``.
        ValueError
            If ``threshold`` is outside ``[0, 1]``.

        Returns
        -------
        BinaryImage
            Pixels whose probability reaches ``threshold``.
        """
        self._validate_threshold(threshold)
        return BinaryImage(value=self.probability_of(class_name) >= threshold)

    def class_map(self, threshold: float = 0.5) -> IntArray:
        """
        Most probable class of every pixel.

        Parameters
        ----------
        threshold : float, optional
            Minimum probability for a pixel to get a class.

        Raises
        ------
        ValueError
            If ``threshold`` is outside ``[0, 1]``.

        Returns
        -------
        IntArray
            Class id of each pixel, ``-1`` where no class reaches ``threshold``, shape (H, W).
        """
        self._validate_threshold(threshold)
        if not len(self.class_names):
            return np.full((self.image_size.height, self.image_size.width), -1, dtype=np.int64)
        best_class_ids: IntArray = self.class_probabilities.argmax(axis=0).astype(np.int64)
        is_assigned: BoolArray = self.class_probabilities.max(axis=0) >= threshold
        return np.where(is_assigned, best_class_ids, -1).astype(np.int64)

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]. got {threshold}")
