pip install -r requirements.txt

wandb init
python EntropySetup.py build_ext --inplace
pip install .
# python -m src.scripts.train --config_file configs/train/base_ae.yml