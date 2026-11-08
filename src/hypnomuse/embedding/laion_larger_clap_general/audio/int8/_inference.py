import os

import numpy as np
import onnxruntime

DEFAULT_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "model/model.onnx")


def load_onnx_model(
    model_path: str | os.PathLike[str] = DEFAULT_OUTPUT_FILE,
) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(os.fspath(model_path))


def get_audio_features(
    model: onnxruntime.InferenceSession,
    input_features: np.ndarray,
    is_longer: np.ndarray,
) -> np.ndarray:
    inputs = {
        "input_features": input_features,
        "is_longer": is_longer,
    }
    input_names = {item.name for item in model.get_inputs()}
    missing_inputs = input_names - inputs.keys()
    if missing_inputs:
        raise ValueError(f"Missing ONNX model inputs: {sorted(missing_inputs)}")

    output_names = [item.name for item in model.get_outputs()]
    try:
        output_index = output_names.index("audio_features")
    except ValueError as error:
        raise ValueError("ONNX model does not expose an audio_features output") from error

    output = model.run(output_names, {name: inputs[name] for name in input_names})
    return output[output_index]
