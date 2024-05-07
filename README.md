# Учебный проект, посвященный сжатию изображений при помощи нейронных сетей

## Запуск в Colab

Файл - run_colab.ipynb

## Запуск локально

run.sh - установка и настройка

### Запуск тренировки с выбранным конфигом
```
python -m src.scripts.train --config_file configs/train/residual_ae.yaml
```

### Запуск инференса с выбранным конфигом
```
python -B -m src.scripts.inference --config_file ./configs/inference/residual_ae.yaml
```

### Модель и результаты на тесте
artifacts/residual_ae-b_2-lr_3e04-b_s_2/epoch_139