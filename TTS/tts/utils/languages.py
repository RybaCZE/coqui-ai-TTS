from typing import Any

import numpy as np
import torch
from coqpit import Coqpit

from TTS.config import get_from_config_or_model_args
from TTS.tts.utils.managers import BaseIDManager


class LanguageManager(BaseIDManager):
    """Manage the languages for multi-lingual 🐸TTS models.

    Args:
        id_file_path: Path to the metafile that maps language names to ids used by TTS models.
        config (Coqpit): Coqpit config that contains the language information in the datasets filed.

    Examples:
        >>> manager = LanguageManager("language_ids.json")
        >>> language_id_mapper = manager.language_ids
    """

    @property
    def num_languages(self) -> int:
        return len(list(self.name_to_id.keys()))

    @property
    def language_names(self) -> list[str]:
        return list(self.name_to_id.keys())

    @staticmethod
    def parse_language_ids_from_config(c: Coqpit) -> dict[str, int]:
        """Set language id from config.

        Args:
            c (Coqpit): Config

        Returns:
            Tuple[Dict, int]: Language ID mapping and the number of languages.
        """
        languages = set({})
        for dataset in c.datasets:
            if "language" in dataset:
                languages.add(dataset["language"])
            else:
                raise ValueError(f"Dataset {dataset['name']} has no language specified.")
        return {name: i for i, name in enumerate(sorted(languages))}

    @staticmethod
    def parse_ids_from_data(items: list[dict[str, Any]], parse_key: str) -> Any:
        raise NotImplementedError

    def set_ids_from_data(self, items: list[dict[str, Any]], parse_key: str) -> Any:
        raise NotImplementedError

    @staticmethod
    def init_from_config(config: Coqpit) -> "LanguageManager":
        """Initialize the language manager from a Coqpit config.

        Args:
            config (Coqpit): Coqpit config.
        """
        if path := get_from_config_or_model_args(config, "language_ids_file"):
            return LanguageManager(path)
        # Fall back to parse language IDs from datasets listed in the config
        language_manager = LanguageManager()
        language_manager.name_to_id = LanguageManager.parse_language_ids_from_config(config)
        return language_manager


def get_language_balancer_weights(items: list[dict[str, Any]]) -> torch.Tensor:
    language_names = np.array([item["language"] for item in items])
    unique_language_names = np.unique(language_names).tolist()
    language_ids = [unique_language_names.index(l) for l in language_names]
    language_count = np.array([len(np.where(language_names == l)[0]) for l in unique_language_names])
    weight_language = 1.0 / language_count
    # get weight for each sample
    dataset_samples_weight = np.array([weight_language[l] for l in language_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
