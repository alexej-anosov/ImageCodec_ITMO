pip install requirements.txt

wandb init
python EntropySetup.py build_ext --inplace
pip install .
python scripts/train.py --config_file configs/train/base_ae.yml