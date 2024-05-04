import os

import click
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from yaml import CLoader

from src.data.make_dataset import ImageDataset
from src.modeling.get_model import load_model
from src.utils import (display_images_and_save_pdf, process_images,
                       set_random_seed)

tqdm.pandas()


@click.command()
@click.option("--config_file", default="config.yaml", help="Path to config YAML file")
def main(config_file):
    with open(config_file, "r") as f:
        args_config = yaml.load(f, Loader=CLoader)

    set_random_seed(args_config["training_args"]["seed"])

    device = (
        "cuda"
        if torch.cuda.is_available() and args_config["training_args"]["use_cuda"]
        else "cpu"
    )

    model = load_model(args_config["model"], model_path=args_config["model_dir"])
    model.to(device)

    b = args_config["b"]
    output_filename = args_config["output_filename"]

    test_dataset = ImageDataset(args_config["data_path"])
    test_loader = DataLoader(
        test_dataset, batch_size=args_config["batch_size"], shuffle=False
    )

    imgs_decoded, imgsQ_decoded, bpp = process_images(test_loader, model, device, b)
    fig, psnr_decoded, psnr_decoded_q, _ = display_images_and_save_pdf(test_dataset, imgs_decoded, imgsQ_decoded, bpp, filepath=output_filename)

if __name__ == "__main__":
    main()
