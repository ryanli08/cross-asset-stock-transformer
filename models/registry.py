from models.baseline_transformer import TransformerBaseline
from models.multi_asset_transformer import MultiAssetTransformer
from models.cross_asset_transformer import CrossAssetTransformer


MODEL_REGISTRY = {
    "transformer_baseline": TransformerBaseline,
    "multi_asset_transformer_baseline": MultiAssetTransformer,
    "cross_asset_transformer": CrossAssetTransformer,
}


def get_model(name: str):
    return MODEL_REGISTRY[name]
