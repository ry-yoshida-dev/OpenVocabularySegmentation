from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field

from open_vocabulary_detector import ImageSize, Prompt

from .box_exemplars import BoxExemplars


@dataclass(frozen=True, eq=False)
class ImageExemplars:
    """
    Box exemplars of one image, keyed by the class of the prompt they define.

    Exemplars are specific to the image they are drawn on, so ``segment_images`` takes one ``ImageExemplars`` (or
    ``None``) per image. Classes without exemplars are segmented from their text queries alone.

    Attributes
    ----------
    class_exemplars : dict[str, BoxExemplars]
        Exemplars keyed by output class name; surrounding whitespace is stripped.

    Raises
    ------
    ValueError
        If there is no class, or a class name is blank or given twice after stripping.
    """

    exemplars_by_class: InitVar[Mapping[str, BoxExemplars]]
    class_exemplars: dict[str, BoxExemplars] = field(init=False)

    def __post_init__(self, exemplars_by_class: Mapping[str, BoxExemplars]) -> None:
        if not exemplars_by_class:
            raise ValueError("image exemplars need at least one class.")
        class_exemplars: dict[str, BoxExemplars] = {}
        for class_name, exemplars in exemplars_by_class.items():
            stripped_name: str = class_name.strip()
            if not stripped_name:
                raise ValueError(f"class names must not be blank. got {list(exemplars_by_class)}")
            if stripped_name in class_exemplars:
                raise ValueError(f"class names must be unique. got {list(exemplars_by_class)}")
            class_exemplars[stripped_name] = exemplars
        object.__setattr__(self, "class_exemplars", class_exemplars)

    def of_class(self, class_name: str) -> BoxExemplars | None:
        """
        Exemplars of one class.

        Parameters
        ----------
        class_name : str
            Output class name.

        Returns
        -------
        BoxExemplars | None
            Exemplars of the class; ``None`` if the class has none in this image.
        """
        return self.class_exemplars.get(class_name.strip())

    def validate_for(self, prompt: Prompt, image_size: ImageSize) -> None:
        """
        Check that the exemplars refer to classes of a prompt and lie on an image.

        Parameters
        ----------
        prompt : Prompt
            Prompt being segmented.
        image_size : ImageSize
            Size of the image the exemplars are drawn on.

        Raises
        ------
        KeyError
            If a class is not a class of the prompt.
        ValueError
            If a box lies outside the image.
        """
        for class_name, exemplars in self.class_exemplars.items():
            if class_name not in prompt.class_names:
                raise KeyError(f"{class_name!r} is not a class of the prompt. classes: {list(prompt.class_names)}")
            exemplars.validate_for(image_size)
