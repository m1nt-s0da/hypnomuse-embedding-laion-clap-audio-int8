from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime
import torch

import hypnomuse.embedding.laion_larger_clap_general.audio.int8._convert as convert_module


class FakeClapModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 2)

    def get_audio_features(
        self,
        *,
        input_features: torch.Tensor,
        is_longer: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert return_dict
        assert is_longer.dtype == torch.bool
        features = input_features.mean(dim=(1, 2))
        features += is_longer.to(input_features.dtype).reshape(-1, 1)
        return SimpleNamespace(pooler_output=self.projection(features))


def test_convert_exports_and_quantizes_audio_features(tmp_path, monkeypatch):
    example_inputs = {
        "input_features": torch.ones((1, 1, 7, 3)),
        "is_longer": torch.zeros((1, 1), dtype=torch.bool),
    }

    class FakeFeatureExtractor:
        nb_max_samples = 8
        sampling_rate = 8

        def __call__(self, *args, **kwargs):
            return example_inputs

    monkeypatch.setattr(
        convert_module.ClapModel,
        "from_pretrained",
        lambda model_id: FakeClapModel(),
    )
    monkeypatch.setattr(
        convert_module.ClapFeatureExtractor,
        "from_pretrained",
        lambda model_id: FakeFeatureExtractor(),
    )

    output_path = tmp_path / "nested" / "audio.onnx"
    convert_module.convert(output=str(output_path))

    onnx_model = onnx.load(output_path)
    assert any(
        initializer.data_type == onnx.TensorProto.INT8
        for initializer in onnx_model.graph.initializer
    )

    session = onnxruntime.InferenceSession(str(output_path))
    result = session.run(
        ["audio_features"],
        {
            "input_features": np.ones((1, 1, 7, 3), dtype=np.float32),
            "is_longer": np.zeros((1, 1), dtype=np.bool_),
        },
    )[0]
    assert result.shape == (1, 2)

    batched_result = session.run(
        ["audio_features"],
        {
            "input_features": np.ones((2, 1, 7, 3), dtype=np.float32),
            "is_longer": np.zeros((2, 1), dtype=np.bool_),
        },
    )[0]
    assert batched_result.shape == (2, 2)
