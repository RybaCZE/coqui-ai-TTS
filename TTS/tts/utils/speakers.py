import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from coqpit import Coqpit

from TTS.config import get_from_config_or_model_args
from TTS.tts.utils.managers import EmbeddingManager

logger = logging.getLogger(__name__)


class SpeakerManager(EmbeddingManager):
    """Manage the speakers for multi-speaker 🐸TTS models.

    Load a datafile and parse the information in a way that can be queried by speaker or clip.

    There are 3 different scenarios considered:

    1. Models using speaker embedding layers. The datafile only maps speaker names to ids used by the embedding layer.
    2. Models using d-vectors. The datafile includes a dictionary in the following format.

    ::

        {
            'clip_name.wav':{
                'name': 'speakerA',
                'embedding'[<d_vector_values>]
            },
            ...
        }


    3. Computing the d-vectors by the speaker encoder. It loads the speaker encoder model and
    computes the d-vectors for a given clip or speaker.

    Args:
        d_vectors_file_path (str, optional): Path to the metafile including x vectors. Defaults to "".
        speaker_id_file_path (str, optional): Path to the metafile that maps speaker names to ids used by
        TTS models. Defaults to "".
        encoder_model_path (str, optional): Path to the speaker encoder model file. Defaults to "".
        encoder_config_path (str, optional): Path to the spealer encoder config file. Defaults to "".

    Examples:
        >>> # load audio processor and speaker encoder
        >>> ap = AudioProcessor(**config.audio)
        >>> manager = SpeakerManager(encoder_model_path=encoder_model_path, encoder_config_path=encoder_config_path)
        >>> # load a sample audio and compute embedding
        >>> waveform = ap.load_wav(sample_wav_path)
        >>> mel = ap.melspectrogram(waveform)
        >>> d_vector = manager.compute_embeddings(mel.T)

    """

    def __init__(
        self,
        *,
        d_vectors_file_path: str = "",
        speaker_id_file_path: str | os.PathLike[Any] = "",
        encoder_model_path: str | os.PathLike[Any] = "",
        encoder_config_path: str | os.PathLike[Any] = "",
        use_cuda: bool = False,
    ) -> None:
        super().__init__(
            embedding_file_path=d_vectors_file_path,
            id_file_path=speaker_id_file_path,
            encoder_model_path=encoder_model_path,
            encoder_config_path=encoder_config_path,
            use_cuda=use_cuda,
        )

    @property
    def num_speakers(self) -> int:
        return len(self.name_to_id)

    @property
    def speaker_names(self) -> list[str]:
        return list(self.name_to_id.keys())

    @staticmethod
    def init_from_config(config: Coqpit) -> "SpeakerManager":
        """Initialize a speaker manager from a config.

        Args:
            config (Coqpit): Config object.

        """
        speaker_manager = SpeakerManager()
        if get_from_config_or_model_args(config, "use_speaker_embedding"):
            if speaker_file := get_from_config_or_model_args(config, "speaker_file"):
                speaker_manager = SpeakerManager(speaker_id_file_path=speaker_file)
            if speakers_file := get_from_config_or_model_args(config, "speakers_file"):
                speaker_manager = SpeakerManager(speaker_id_file_path=speakers_file)
        elif get_from_config_or_model_args(config, "use_d_vector_file"):
            if d_vector_file := get_from_config_or_model_args(config, "d_vector_file"):
                speaker_manager = SpeakerManager(d_vectors_file_path=d_vector_file)

        se_model_path = get_from_config_or_model_args(config, "speaker_encoder_model_path", "")
        se_config_path = get_from_config_or_model_args(config, "speaker_encoder_config_path", "")

        if Path(se_model_path).is_file() and Path(se_config_path).is_file():
            speaker_manager.init_encoder(se_model_path, se_config_path)
        return speaker_manager


def get_speaker_balancer_weights(items: list[dict[str, Any]]) -> torch.Tensor:
    speaker_names = np.array([item["speaker_name"] for item in items])
    unique_speaker_names = np.unique(speaker_names).tolist()
    speaker_ids = [unique_speaker_names.index(spk) for spk in speaker_names]
    speaker_count = np.array([len(np.where(speaker_names == spk)[0]) for spk in unique_speaker_names])
    weight_speaker = 1.0 / speaker_count
    dataset_samples_weight = np.array([weight_speaker[i] for i in speaker_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
