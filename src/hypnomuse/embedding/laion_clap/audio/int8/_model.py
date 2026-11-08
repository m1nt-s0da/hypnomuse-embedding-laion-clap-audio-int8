from ._model_loaders import load_feature_extractor
from ._inference import get_audio_features, load_onnx_model
import numpy as np
from typing_extensions import deprecated

__all__ = ["LaionClapAudioInt8", "LaionLargerClapGeneralAudioInt8"]


class LaionClapAudioInt8:
    def __init__(self):
        self._feature_extractor = load_feature_extractor()
        self._model = load_onnx_model()

    def prepare_audio_inputs(
        self, waveform: np.ndarray, *, sampling_rate: int | None = None
    ):
        inputs = self._feature_extractor(
            waveform,
            sampling_rate=sampling_rate,
            return_tensors="np",
        )
        return inputs["input_features"], inputs["is_longer"]

    def get_audio_features(self, input_features: np.ndarray, is_longer: np.ndarray):
        return get_audio_features(self._model, input_features, is_longer)

    def __call__(self, waveform: np.ndarray, *, sampling_rate: int | None = None):
        input_feature, is_longer = self.prepare_audio_inputs(
            waveform, sampling_rate=sampling_rate
        )
        return get_audio_features(self._model, input_feature, is_longer)


@deprecated("Use LaionClapAudioInt8 instead")
class LaionLargerClapGeneralAudioInt8(LaionClapAudioInt8): ...
