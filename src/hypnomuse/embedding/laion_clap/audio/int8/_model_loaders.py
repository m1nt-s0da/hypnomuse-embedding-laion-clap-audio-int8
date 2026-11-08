from transformers import ClapFeatureExtractor
import os

model_dir = os.path.join(os.path.dirname(__file__), "model")


def load_feature_extractor():
    return ClapFeatureExtractor.from_pretrained(model_dir)
