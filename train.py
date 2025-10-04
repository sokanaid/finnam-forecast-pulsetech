"""
Скрипт для обучения модели FORECAST
"""
import sys
import time
import argparse
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.model import ForecastModel
from src.config import RANDOM_SEED, MODELS_DIR, TRAIN_CANDLES, TRAIN_NEWS

import numpy as np
import pandas as pd
np.random.seed(RANDOM_SEED)


def main(train_candles_path=None, train_news_path=None, output_dir=None):
    """Основная функция обучения"""
    start_time = time.time()
    
    # Используем пути по умолчанию если не указаны
    train_candles_path = train_candles_path or TRAIN_CANDLES
    train_news_path = train_news_path or TRAIN_NEWS
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 ОБУЧЕНИЕ МОДЕЛИ FORECAST")
    print("=" * 70)
    print(f"\n📁 Входные данные:")
    print(f"   Train candles: {train_candles_path}")
    print(f"   Train news: {train_news_path}")
    print(f"   Output dir: {output_dir}")
    
    # 1. Загрузка данных
    print("\n" + "=" * 70)
    print("ЭТАП 1: Загрузка данных")
    print("=" * 70)
    
    # Временно переопределяем пути в config (для DataLoader)
    import src.config as config
    config.TRAIN_CANDLES = train_candles_path
    config.TRAIN_NEWS = train_news_path
    
    loader = DataLoader()
    loader.load_data(mode='train')
    
    train_df = loader.get_train_data()
    all_candles = loader.get_all_candles()
    all_news = loader.get_all_news()
    
    print(f"\n📊 Статистика данных:")
    print(f"   Тикеров: {train_df['ticker'].nunique()}")
    print(f"   Дат: {train_df['begin'].nunique()}")
    print(f"   Период: {train_df['begin'].min()} - {train_df['begin'].max()}")
    
    # 2. Подготовка таргетов для всех горизонтов
    print("\n" + "=" * 70)
    print("ЭТАП 2: Подготовка таргетов")
    print("=" * 70)
    
    all_candles = loader.prepare_targets_for_horizons(all_candles)
    
    # Фильтруем только train данные
    train_dates = set(train_df['begin'].unique())
    train_data = all_candles[all_candles['begin'].isin(train_dates)].copy()
    
    print(f"   Строк для обучения: {len(train_data)}")
    
    # 3. Feature Engineering
    print("\n" + "=" * 70)
    print("ЭТАП 3: Feature Engineering")
    print("=" * 70)
    
    engineer = FeatureEngineer()
    
    # Создаем признаки из котировок
    all_candles = engineer.create_price_features(all_candles)
    
    # Создаем признаки из новостей (упрощенная версия для экономии времени)
    print("\n   ⏭️  Пропускаем детальную обработку новостей (занимает много времени)")
    print("   Создаем базовые агрегации новостей...")
    
    # Быстрые агрегации новостей по дням
    news_per_day = all_news.groupby(
        all_news['publish_date'].dt.date
    ).size().reset_index(name='daily_news_count')
    news_per_day.columns = ['date', 'daily_news_count']
    news_per_day['date'] = pd.to_datetime(news_per_day['date'])
    
    # Добавляем к котировкам
    all_candles['date_only'] = all_candles['begin'].dt.date
    all_candles['date_only'] = pd.to_datetime(all_candles['date_only'])
    all_candles = all_candles.merge(
        news_per_day, 
        left_on='date_only', 
        right_on='date', 
        how='left'
    )
    all_candles['daily_news_count'] = all_candles['daily_news_count'].fillna(0)
    all_candles = all_candles.drop(['date_only', 'date'], axis=1)
    
    # Заполняем пропущенные значения
    all_candles = engineer.fill_missing_values(all_candles)
    
    # Получаем список признаков
    feature_cols = engineer.get_feature_columns(all_candles)
    feature_cols.append('daily_news_count')  # Добавляем новостную статистику
    
    print(f"\n   ✅ Всего признаков: {len(feature_cols)}")
    
    # Фильтруем только train данные
    train_data = all_candles[all_candles['begin'].isin(train_dates)].copy()
    
    # 4. Обучение модели
    print("\n" + "=" * 70)
    print("ЭТАП 4: Обучение модели")
    print("=" * 70)
    
    model = ForecastModel()
    model.train(train_data, feature_cols, val_size=0.2, verbose=True)
    
    # 5. Сохранение модели
    print("\n" + "=" * 70)
    print("ЭТАП 5: Сохранение модели")
    print("=" * 70)
    
    model.save(output_dir / "forecast_models.pkl")
    
    # Сохраняем также обработанные данные для предсказания
    print("\n💾 Сохранение обработанных данных...")
    processed_data = {
        'all_candles': all_candles,
        'feature_cols': feature_cols,
        'news_per_day': news_per_day
    }
    
    import joblib
    joblib.dump(processed_data, output_dir / "processed_data.pkl")
    print(f"   ✓ Данные сохранены")
    
    # Время выполнения
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 70)
    print(f"⏱️  Время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")
    print(f"💾 Модели сохранены в: {output_dir}")
    print("\n💡 Следующий шаг: запустите predict.py для создания предсказаний")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обучение модели FORECAST')
    parser.add_argument('--train-candles', type=str, default=None,
                       help=f'Путь к файлу с обучающими котировками (по умолчанию: {TRAIN_CANDLES})')
    parser.add_argument('--train-news', type=str, default=None,
                       help=f'Путь к файлу с обучающими новостями (по умолчанию: {TRAIN_NEWS})')
    parser.add_argument('--output-dir', type=str, default=None,
                       help=f'Директория для сохранения моделей (по умолчанию: {MODELS_DIR})')
    
    args = parser.parse_args()
    
    main(
        train_candles_path=args.train_candles,
        train_news_path=args.train_news,
        output_dir=args.output_dir
    )

