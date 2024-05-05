from src.modeling.base_ae import BaseAutoEncoder
from src.modeling.ln_gelu_ae import LNAutoEncoder
from src.modeling.xavier_ln_gelu_ae import XAvierLNAutoEncoder

MODEL_DICT = {
    "base_ae": BaseAutoEncoder,
    "ln_ae": LNAutoEncoder,
    "xv_ln_ae": XAvierLNAutoEncoder,
}


def init_model(model_cfg):
    model_kwargs = (
        {} if model_cfg["model_kwargs"] is None else model_cfg["model_kwargs"]
    )
    model = MODEL_DICT[model_cfg["type"]](model_cfg["model_name"], **model_kwargs)

    return model


def load_model(model_cfg, model_path, device):
    model_kwargs = (
        {} if model_cfg["model_kwargs"] is None else model_cfg["model_kwargs"]
    )

    model = MODEL_DICT[model_cfg["type"]](model_cfg["model_name"], **model_kwargs)
    model.load(device=device, directory=model_path)

    return model
