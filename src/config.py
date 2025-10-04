"""
Конфигурация для задачи FORECAST
"""
import os
from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Создаем директории если их нет
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Файлы данных
TRAIN_CANDLES = DATA_RAW / "train_candles.csv"
TRAIN_NEWS = DATA_RAW / "train_news.csv"
PUBLIC_TEST_CANDLES = DATA_RAW / "public_test_candles.csv"
PRIVATE_TEST_CANDLES = DATA_RAW / "private_test_candles.csv"
TEST_NEWS = DATA_RAW / "test_news.csv"

# Гиперпараметры
RANDOM_SEED = 42
WINDOW_SIZE = 20  # Размер скользящего окна для признаков
N_HORIZONS = 20  # Количество горизонтов прогнозирования

# Параметры модели
MODEL_PARAMS = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'verbose': 0
}

# Параметры NLP
MAX_NEWS_LENGTH = 512  # Максимальная длина текста новости
NEWS_EMBEDDING_DIM = 384  # Размерность эмбеддингов новостей (для sentence-transformers)

# Параметры обучения
TRAIN_VAL_SPLIT = 0.8  # Доля train в train/val split
MIN_TRAIN_DATE = "2020-06-19"  # Начальная дата обучения

