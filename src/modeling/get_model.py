from src.modeling.base_ae import BaseAutoEncoder

MODEL_DICT = {"base_ae": BaseAutoEncoder}


def init_model(model_cfg):
    model = MODEL_DICT[model_cfg["type"]](
        model_cfg["model_name"], **model_cfg["model_kwargs"]
    )

    return model


def load_model(model_cfg, model_path):
    model = MODEL_DICT[model_cfg["type"]](
        model_cfg["model_name"], **model_cfg["model_kwargs"]
    )
    model.load(directory=model_path)

    return model
