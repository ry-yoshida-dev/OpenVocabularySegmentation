from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from open_vocabulary_detector import ImageSize, Prompt

from ...array_types import BoolArray, Float32Array, FloatArray, IntArray
from ...result import SegmentationResult, SemanticSegmentationResult


@dataclass(frozen=True, eq=False)
class Sam3Prediction:
    """
    Accumulates the instances and semantic probabilities of a mini-batch while SAM 3 runs query by query.

    Attributes
    ----------
    prompt : Prompt
        Prompt being segmented.
    image_sizes : tuple[ImageSize, ...]
        Size of each image of the mini-batch.
    is_semantic_requested : bool
        Whether semantic probabilities are collected.
    instance_masks : tuple[list[BoolArray], ...]
        Instance masks of each image, one array per query run.
    instance_confidences : tuple[list[FloatArray], ...]
        Instance confidences of each image, one array per query run.
    instance_query_ids : tuple[list[IntArray], ...]
        Matched query of each instance of each image, one array per query run.
    class_probabilities : tuple[Float32Array, ...]
        Per-class probability maps of each image, shape (C, H, W); empty arrays when semantics are not requested.
    """

    prompt: Prompt
    image_sizes: tuple[ImageSize, ...]
    is_semantic_requested: bool
    instance_masks: tuple[list[BoolArray], ...]
    instance_confidences: tuple[list[FloatArray], ...]
    instance_query_ids: tuple[list[IntArray], ...]
    class_probabilities: tuple[Float32Array, ...]

    @classmethod
    def start(cls, prompt: Prompt, image_sizes: Sequence[ImageSize], is_semantic_requested: bool) -> "Sam3Prediction":
        """
        Create an empty accumulator.

        Parameters
        ----------
        prompt : Prompt
            Prompt being segmented.
        image_sizes : Sequence[ImageSize]
            Size of each image of the mini-batch.
        is_semantic_requested : bool
            Whether semantic probabilities are collected.

        Returns
        -------
        Sam3Prediction
            Accumulator without instances and with zero probabilities.
        """
        class_count: int = len(prompt.class_names) if is_semantic_requested else 0
        return cls(
            prompt=prompt,
            image_sizes=tuple(image_sizes),
            is_semantic_requested=is_semantic_requested,
            instance_masks=tuple([] for _ in image_sizes),
            instance_confidences=tuple([] for _ in image_sizes),
            instance_query_ids=tuple([] for _ in image_sizes),
            class_probabilities=tuple(
                np.zeros((class_count, size.height, size.width) if class_count else (0, 0, 0), dtype=np.float32)
                for size in image_sizes
            ),
        )

    def add_instances(self, image_index: int, masks: BoolArray, confidences: FloatArray, query_ids: IntArray) -> None:
        """
        Record the instances one query run found in one image.

        Parameters
        ----------
        image_index : int
            Index of the image in the mini-batch.
        masks : BoolArray
            Instance masks, shape (N, H, W).
        confidences : FloatArray
            Instance confidences, shape (N,).
        query_ids : IntArray
            Matched query of each instance, shape (N,).
        """
        self.instance_masks[image_index].append(masks)
        self.instance_confidences[image_index].append(confidences)
        self.instance_query_ids[image_index].append(query_ids)

    def add_semantic(self, image_index: int, class_id: int, probabilities: Float32Array) -> None:
        """
        Merge the semantic probabilities of one query run into its class by maximum.

        Parameters
        ----------
        image_index : int
            Index of the image in the mini-batch.
        class_id : int
            Class of the query.
        probabilities : Float32Array
            Probability map of the query, shape (H, W).

        Raises
        ------
        ValueError
            If semantic probabilities were not requested.
        """
        if not self.is_semantic_requested:
            raise ValueError("semantic probabilities were not requested.")
        class_map: Float32Array = self.class_probabilities[image_index][class_id]
        np.maximum(class_map, probabilities, out=class_map)

    @property
    def instances(self) -> tuple[SegmentationResult, ...]:
        """
        Raw instances of every image, before the shared post-processing.

        Returns
        -------
        tuple[SegmentationResult, ...]
            One result per image, in mini-batch order.
        """
        return tuple(
            SegmentationResult.from_masks(
                masks=np.concatenate(masks) if masks else np.zeros((0, size.height, size.width), dtype=np.bool_),
                confidences=np.concatenate(confidences) if confidences else np.zeros((0,), dtype=np.float64),
                query_ids=np.concatenate(query_ids) if query_ids else np.zeros((0,), dtype=np.int64),
                prompt=self.prompt,
                image_size=size,
            )
            for masks, confidences, query_ids, size in zip(
                self.instance_masks, self.instance_confidences, self.instance_query_ids, self.image_sizes, strict=True
            )
        )

    @property
    def semantics(self) -> tuple[SemanticSegmentationResult, ...]:
        """
        Per-class probabilities of every image.

        Raises
        ------
        ValueError
            If semantic probabilities were not requested.

        Returns
        -------
        tuple[SemanticSegmentationResult, ...]
            One result per image, in mini-batch order.
        """
        if not self.is_semantic_requested:
            raise ValueError("semantic probabilities were not requested.")
        return tuple(
            SemanticSegmentationResult(class_probabilities=probabilities, prompt=self.prompt, image_size=size)
            for probabilities, size in zip(self.class_probabilities, self.image_sizes, strict=True)
        )
