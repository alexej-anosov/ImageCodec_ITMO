pip install -r requirements.txt

wandb init
python EntropySetup.py build_ext --inplace
pip install .
