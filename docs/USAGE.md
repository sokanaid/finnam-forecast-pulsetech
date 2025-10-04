# Примеры использования FORECAST

## 🚀 Быстрый старт (стандартные пути)

```bash
# Шаг 1: Обучение модели
python train.py

# Шаг 2: Создание предсказаний
python predict.py

# Результат: submission.csv готов!
```

## 📂 Работа с разными файлами

### Обучение на альтернативных данных

```bash
python train.py \
  --train-candles /path/to/my_train_candles.csv \
  --train-news /path/to/my_train_news.csv \
  --output-dir ./my_models/
```

### Создание предсказаний с другими моделями

```bash
python predict.py \
  --public-test-candles /path/to/public_test.csv \
  --private-test-candles /path/to/private_test.csv \
  --test-news /path/to/test_news.csv \
  --models-dir ./my_models/ \
  --output my_submission.csv
```

## 🔍 Получение справки

```bash
# Справка по обучению
python train.py --help

# Справка по предсказанию
python predict.py --help

# Проверка submission
python evaluation.py --help
```

## 📊 Проверка результатов

```bash
# Проверка формата и статистики
python evaluation.py submission.csv

# Проверка кастомного файла
python evaluation.py my_submission.csv
```

## 🎯 Типичные сценарии

### Сценарий 1: Эксперимент с новыми данными

```bash
# 1. Подготовить данные в директории experiments/data1/
# 2. Обучить модель
python train.py \
  --train-candles experiments/data1/train_candles.csv \
  --train-news experiments/data1/train_news.csv \
  --output-dir experiments/models1/

# 3. Протестировать
python predict.py \
  --models-dir experiments/models1/ \
  --output experiments/submission1.csv
```

### Сценарий 2: Несколько версий моделей

```bash
# Версия 1
python train.py --output-dir models/v1/
python predict.py --models-dir models/v1/ --output submission_v1.csv

# Версия 2 (с другими данными)
python train.py \
  --train-candles data/augmented/train_candles.csv \
  --output-dir models/v2/
python predict.py --models-dir models/v2/ --output submission_v2.csv

# Сравнение результатов
python evaluation.py submission_v1.csv
python evaluation.py submission_v2.csv
```

### Сценарий 3: Отладка на подмножестве данных

```bash
# Использовать первые N строк для быстрой проверки
head -1000 data/raw/train_candles.csv > data/debug/train_candles_small.csv
head -500 data/raw/train_news.csv > data/debug/train_news_small.csv

# Быстрое обучение
python train.py \
  --train-candles data/debug/train_candles_small.csv \
  --train-news data/debug/train_news_small.csv \
  --output-dir models/debug/
```

## ⚙️ Структура директорий для экспериментов

```
finnam_hackaton/
├── data/
│   ├── raw/              # Исходные данные
│   ├── augmented/        # Дополненные данные
│   └── debug/            # Данные для отладки
├── models/
│   ├── v1/               # Версия 1
│   ├── v2/               # Версия 2
│   └── debug/            # Отладочные модели
├── experiments/
│   ├── exp1/             # Эксперимент 1
│   └── exp2/             # Эксперимент 2
└── submissions/
    ├── submission_v1.csv
    ├── submission_v2.csv
    └── final_submission.csv
```

## 💡 Полезные команды

```bash
# Посмотреть структуру CSV файла
head -5 data/raw/train_candles.csv

# Посчитать строки
wc -l data/raw/*.csv

# Проверить размер моделей
ls -lh models/*.pkl

# Быстро посмотреть submission
head -10 submission.csv
tail -10 submission.csv
```

