def test_load_feature_extractor():
    from hypnomuse.embedding.laion_clap.audio.int8._model_loaders import (
        load_feature_extractor,
    )

    feature_extractor = load_feature_extractor()
    assert feature_extractor is not None
