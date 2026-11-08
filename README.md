# CLAP audio embedding (ONNX INT8)

Convert the CLAP audio feature model to a dynamically quantized INT8 ONNX model:

```shell
uv run convert
```

By default, the converter uses `laion/clap-htsat-unfused` and writes
`model.onnx` under `src/hypnomuse/embedding/clap/audio/int8/model/`.
Specify a different model or output path with `--model_id` and `--output`:

```shell
uv run convert --model_id laion/clap-htsat-unfused --output model.onnx
```

The ONNX model accepts `input_features` and `is_longer` and returns the
normalized audio embedding from `ClapModel.get_audio_features`.

`get_audio_features` in `hypnomuse.embedding.clap.audio.int8.inference`
runs the exported model with ONNX Runtime. To run the audio integration test,
set `TEST_AUDIO_FILES` in `.env` to an audio file path, or to multiple file
paths joined by `os.path.sep`. The test skips when the variable is empty.

Compare INT8 ONNX output with the original float CLAP model on the same audio:

```shell
uv run pytest -s tests/test_inference.py::test_int8_onnx_error_against_float_model
```

The test reports MAE, RMSE, maximum absolute error, cosine similarity, and
relative L2 error for each audio file, and records the measurements as pytest
properties. It checks matching output shapes and finite values without imposing
an arbitrary accuracy threshold.

## Licensing

The library source is MIT licensed; the model files under
`src/hypnomuse/embedding/laion_larger_clap_general/audio/int8/model/` are
derived from `laion/larger_clap_general` and are distributed under the
upstream model card's Apache-2.0 license. See [LICENSE](LICENSE) and
[NOTICE](NOTICE); the model directory contains the Apache-2.0 license text.