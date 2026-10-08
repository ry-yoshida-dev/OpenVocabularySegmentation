from typing import cast

from transformers import AutoConfig, PretrainedConfig


class CheckpointConfig:
    """
    Reads the configuration of a ``transformers`` checkpoint without loading its weights.

    Segmenters check the ``model_type`` stored in the checkpoint's ``config.json`` before loading, so weights of
    another architecture fail with a clear error instead of loading partially.
    """

    @staticmethod
    def read_model_type(weights_path: str) -> str:
        """
        Read the architecture identifier of a checkpoint.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.

        Raises
        ------
        TypeError
            If ``transformers`` does not return a model configuration.

        Returns
        -------
        str
            ``model_type`` of the checkpoint, e.g. ``"sam"`` or ``"sam2_video"``.
        """
        config: object = cast(object, AutoConfig.from_pretrained(weights_path))
        if not isinstance(config, PretrainedConfig):
            raise TypeError(f"expected a model configuration for {weights_path}. got {type(config)}")
        return config.model_type

    @classmethod
    def require_model_type(cls, weights_path: str, model_types: frozenset[str]) -> str:
        """
        Check that a checkpoint has one of the expected architectures.

        Parameters
        ----------
        weights_path : str
            Hugging Face Hub model id or local checkpoint directory.
        model_types : frozenset[str]
            Accepted ``model_type`` values.

        Raises
        ------
        ValueError
            If the checkpoint has another ``model_type``.

        Returns
        -------
        str
            ``model_type`` of the checkpoint.
        """
        model_type: str = cls.read_model_type(weights_path)
        if model_type not in model_types:
            raise ValueError(f"{weights_path} is a {model_type!r} checkpoint. expected one of {sorted(model_types)}")
        return model_type
