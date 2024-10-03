# Учебный проект, посвященный сжатию изображений при помощи нейронных сетей

## Запуск в Colab

Файл - run_kaggle.ipynb

## Запуск локально

run.sh - установка и настройка

### Запуск тренировки с выбранным конфигом
```
python -m src.scripts.train --config_file ./configs/train/base_ae_b32_s6000_lr3e3_orthogonal_bottleneck_softplus.yml
```

