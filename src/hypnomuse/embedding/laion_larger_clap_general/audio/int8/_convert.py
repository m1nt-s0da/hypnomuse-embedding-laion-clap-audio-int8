import os
import tempfile
import warnings

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import ClapFeatureExtractor, ClapModel

DEFAULT_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "model/model.onnx")
DEFAULT_MODEL_ID = "laion/clap-htsat-unfused"


class _AudioFeatures(torch.nn.Module):
    def __init__(self, model: ClapModel):
        super().__init__()
        self.model = model

    def forward(
        self, input_features: torch.Tensor, is_longer: torch.Tensor
    ) -> torch.Tensor:
        output = self.model.get_audio_features(
            input_features=input_features,
            is_longer=is_longer,
            return_dict=True,
        )
        return output.pooler_output


def convert(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    output: str | os.PathLike[str] = DEFAULT_OUTPUT_FILE,
) -> None:
    model = ClapModel.from_pretrained(model_id)
    model.eval()
    feature_extractor = ClapFeatureExtractor.from_pretrained(model_id)
    example_inputs = feature_extractor(
        torch.zeros(feature_extractor.nb_max_samples),
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
    )
    audio_features = _AudioFeatures(model).eval()

    output_path = os.path.abspath(output)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        float_model_path = os.path.join(temp_dir, "model.onnx")
        quantized_model_path = os.path.join(temp_dir, "model.int8.onnx")

        with torch.no_grad():
            with warnings.catch_warnings():
                # The two inputs intentionally share one symbolic batch dimension.
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"# The axis name: .* will not be used, since it shares "
                        r"the same shape constraints with another axis: .*"
                    ),
                    category=UserWarning,
                )
                torch.onnx.export(
                    audio_features,
                    (example_inputs["input_features"], example_inputs["is_longer"]),
                    float_model_path,
                    input_names=["input_features", "is_longer"],
                    output_names=["audio_features"],
                    dynamic_shapes=(
                        {0: torch.export.Dim("input_batch")},
                        {0: torch.export.Dim("longer_batch")},
                    ),
                    opset_version=18,
                )

        onnx_model = onnx.load(float_model_path)
        initializer_names = {
            initializer.name for initializer in onnx_model.graph.initializer
        }
        # ORT transposes Gemm weights before shape inference during quantization.
        retained_value_info = [
            value_info
            for value_info in onnx_model.graph.value_info
            if value_info.name not in initializer_names
        ]
        del onnx_model.graph.value_info[:]
        onnx_model.graph.value_info.extend(retained_value_info)
        onnx.save(onnx_model, float_model_path)

        quantize_dynamic(
            float_model_path,
            quantized_model_path,
            weight_type=QuantType.QInt8,
        )
        os.replace(quantized_model_path, output_path)


def _convert_main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()
    print(args)
    convert(**vars(args))
