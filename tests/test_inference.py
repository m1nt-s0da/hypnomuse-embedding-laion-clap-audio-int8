import os
from pathlib import Path

import numpy as np
import pytest
import torch
from dotenv import load_dotenv
from pydub import AudioSegment
from transformers import ClapModel

from hypnomuse.embedding.laion_clap.audio.int8._model_loaders import (
    load_feature_extractor,
)
from hypnomuse.embedding.laion_clap.audio.int8._inference import (
    get_audio_features,
    load_onnx_model,
)

MODEL_ID = "laion/clap-htsat-unfused"


def _parse_audio_files(value: str) -> list[str]:
    if Path(value).is_file():
        return [value]

    def parse_from(start: int) -> list[str]:
        for end in range(start, len(value) + 1):
            if end < len(value) and value[end] != os.path.sep:
                continue
            path = value[start:end].strip()
            if path and Path(path).is_file():
                if end == len(value):
                    return [path]
                try:
                    return [path, *parse_from(end + 1)]
                except ValueError:
                    continue
        raise ValueError(
            "TEST_AUDIO_FILES must contain existing paths separated by os.path.sep"
        )

    return parse_from(0)


def _extract_audio_inputs(audio_file: str, feature_extractor):
    audio = AudioSegment.from_file(audio_file)
    audio = (
        audio.set_channels(1)
        .set_frame_rate(feature_extractor.sampling_rate)
        .set_sample_width(2)
    )
    waveform = np.asarray(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    inputs = feature_extractor(
        waveform,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="np",
    )
    return inputs["input_features"], inputs["is_longer"]


def test_inference_with_test_audio_files():
    load_dotenv()
    configured_files = os.getenv("TEST_AUDIO_FILES", "").strip()
    if not configured_files:
        pytest.skip("TEST_AUDIO_FILES is empty")

    audio_files = _parse_audio_files(configured_files)
    model = load_onnx_model()
    feature_extractor = load_feature_extractor()

    for audio_file in audio_files:
        input_features, is_longer = _extract_audio_inputs(audio_file, feature_extractor)
        features = get_audio_features(
            model,
            input_features,
            is_longer,
        )

        assert features.shape == (1, 512)
        assert np.isfinite(features).all()


def test_int8_onnx_error_against_float_model(record_property):
    load_dotenv()
    configured_files = os.getenv("TEST_AUDIO_FILES", "").strip()
    if not configured_files:
        pytest.skip("TEST_AUDIO_FILES is empty")

    audio_files = _parse_audio_files(configured_files)
    feature_extractor = load_feature_extractor()
    float_model = ClapModel.from_pretrained(MODEL_ID).eval()
    int8_model = load_onnx_model()
    audio_errors = []

    for index, audio_file in enumerate(audio_files, start=1):
        input_features, is_longer = _extract_audio_inputs(audio_file, feature_extractor)
        with torch.inference_mode():
            float_features = float_model.get_audio_features(
                input_features=torch.from_numpy(input_features),
                is_longer=torch.from_numpy(is_longer),
            ).pooler_output.numpy()

        int8_features = get_audio_features(
            int8_model,
            input_features,
            is_longer,
        )
        assert int8_features.shape == float_features.shape
        assert np.isfinite(float_features).all()
        assert np.isfinite(int8_features).all()

        difference = int8_features - float_features
        absolute_difference = np.abs(difference)
        denominator = np.linalg.norm(float_features, axis=1) * np.linalg.norm(
            int8_features, axis=1
        )
        cosine_similarity = np.sum(float_features * int8_features, axis=1) / denominator
        metrics = {
            "mae": float(np.mean(absolute_difference)),
            "rmse": float(np.sqrt(np.mean(np.square(difference)))),
            "max_absolute_error": float(np.max(absolute_difference)),
            "cosine_similarity": float(np.mean(cosine_similarity)),
            "relative_l2_error": float(
                np.linalg.norm(difference) / np.linalg.norm(float_features)
            ),
        }
        assert np.isfinite(list(metrics.values())).all()
        audio_errors.append(metrics)
        print(f"audio[{index}] INT8 vs float: {metrics}")
        for metric, value in metrics.items():
            record_property(f"int8_vs_float_audio_{index}_{metric}", value)

    for metric in audio_errors[0]:
        record_property(
            f"int8_vs_float_{metric}",
            float(np.mean([item[metric] for item in audio_errors])),
        )


def test_parse_audio_files_with_os_path_separator(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.touch()
    second.touch()

    assert _parse_audio_files(os.path.sep.join((str(first), str(second)))) == [
        str(first),
        str(second),
    ]
