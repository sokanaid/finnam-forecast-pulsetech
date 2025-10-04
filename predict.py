"""
Скрипт для создания предсказаний FORECAST
"""
import sys
import time
import argparse
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.model import ForecastModel
from src.config import MODELS_DIR, PROJECT_ROOT, PUBLIC_TEST_CANDLES, PRIVATE_TEST_CANDLES, TEST_NEWS

import numpy as np
import pandas as pd
import joblib


def main(public_test_candles=None, private_test_candles=None, test_news=None, 
         models_dir=None, output_file=None):
    """Основная функция предсказания"""
    start_time = time.time()
    
    # Используем пути по умолчанию если не указаны
    public_test_candles = public_test_candles or PUBLIC_TEST_CANDLES
    private_test_candles = private_test_candles or PRIVATE_TEST_CANDLES
    test_news = test_news or TEST_NEWS
    models_dir = Path(models_dir) if models_dir else MODELS_DIR
    output_file = output_file or (PROJECT_ROOT / "submission.csv")
    
    print("=" * 70)
    print("🔮 СОЗДАНИЕ ПРЕДСКАЗАНИЙ FORECAST")
    print("=" * 70)
    print(f"\n📁 Входные данные:")
    print(f"   Public test: {public_test_candles}")
    print(f"   Private test: {private_test_candles}")
    print(f"   Test news: {test_news}")
    print(f"   Models dir: {models_dir}")
    print(f"   Output: {output_file}")
    
    # 1. Загрузка модели
    print("\n" + "=" * 70)
    print("ЭТАП 1: Загрузка модели")
    print("=" * 70)
    
    model = ForecastModel()
    model.load(models_dir / "forecast_models.pkl")
    
    # Загружаем обработанные данные
    print("\n📂 Загрузка обработанных данных...")
    processed_data = joblib.load(models_dir / "processed_data.pkl")
    all_candles = processed_data['all_candles']
    feature_cols = processed_data['feature_cols']
    
    print(f"   ✓ Данных загружено: {len(all_candles)} строк")
    print(f"   ✓ Признаков: {len(feature_cols)}")
    
    # 2. Загрузка тестовых данных
    print("\n" + "=" * 70)
    print("ЭТАП 2: Загрузка тестовых данных")
    print("=" * 70)
    
    # Временно переопределяем пути в config (для DataLoader)
    import src.config as config
    config.PUBLIC_TEST_CANDLES = public_test_candles
    config.PRIVATE_TEST_CANDLES = private_test_candles
    config.TEST_NEWS = test_news
    
    loader = DataLoader()
    loader.load_data(mode='test')
    test_df = loader.get_test_data()
    
    print(f"   Тестовых строк: {len(test_df)}")
    print(f"   Тикеров: {test_df['ticker'].nunique()}")
    print(f"   Период: {test_df['begin'].min()} - {test_df['begin'].max()}")
    
    # 3. Подготовка тестовых данных
    print("\n" + "=" * 70)
    print("ЭТАП 3: Подготовка тестовых данных")
    print("=" * 70)
    
    # Фильтруем только тестовые даты из обработанных данных
    test_dates = set(test_df['begin'].unique())
    test_data = all_candles[all_candles['begin'].isin(test_dates)].copy()
    
    print(f"   Строк для предсказания: {len(test_data)}")
    
    # Проверяем наличие всех признаков
    missing_features = set(feature_cols) - set(test_data.columns)
    if missing_features:
        print(f"   ⚠️  Отсутствующие признаки: {missing_features}")
        for feat in missing_features:
            test_data[feat] = 0
    
    # 4. Создание предсказаний
    print("\n" + "=" * 70)
    print("ЭТАП 4: Создание предсказаний")
    print("=" * 70)
    
    predictions = model.predict(test_data)
    
    # 5. Сохранение результатов
    print("\n" + "=" * 70)
    print("ЭТАП 5: Сохранение результатов")
    print("=" * 70)
    
    # Сортируем по ticker и begin для консистентности
    predictions = predictions.sort_values(['ticker', 'begin']).reset_index(drop=True)
    
    # Оставляем только последнюю дату для каждого тикера (самый свежий прогноз)
    predictions = predictions.groupby('ticker').last().reset_index()
    
    # Удаляем колонку begin, оставляем только ticker и вероятности
    output_cols = ['ticker'] + [f'p{i}' for i in range(1, 21)]
    submission = predictions[output_cols].copy()
    
    submission.to_csv(output_file, index=False)
    
    print(f"\n💾 Submission сохранен: {output_file}")
    print(f"   Строк: {len(submission)}")
    print(f"   Колонок: {len(submission.columns)}")
    
    # Показываем примеры
    print(f"\n📋 Первые строки submission:")
    print(submission.head(10).to_string(index=False, max_cols=10))
    
    # Статистика
    prob_cols = [f'p{i}' for i in range(1, 21)]
    print(f"\n📊 Статистика вероятностей:")
    print(f"   Среднее: {submission[prob_cols].mean().mean():.4f}")
    print(f"   Std: {submission[prob_cols].std().mean():.4f}")
    print(f"   Min: {submission[prob_cols].min().min():.4f}")
    print(f"   Max: {submission[prob_cols].max().max():.4f}")
    
    # Распределение по тикерам
    print(f"\n📈 Строк по тикерам:")
    ticker_counts = submission['ticker'].value_counts().sort_index()
    for ticker, count in ticker_counts.items():
        print(f"   {ticker}: {count}")
    
    # Время выполнения
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("✅ ПРЕДСКАЗАНИЯ СОЗДАНЫ!")
    print("=" * 70)
    print(f"⏱️  Время выполнения: {elapsed_time:.2f} секунд ({elapsed_time/60:.2f} минут)")
    print(f"💾 Результат сохранен: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Создание предсказаний FORECAST')
    parser.add_argument('--public-test-candles', type=str, default=None,
                       help=f'Путь к public test котировкам (по умолчанию: {PUBLIC_TEST_CANDLES})')
    parser.add_argument('--private-test-candles', type=str, default=None,
                       help=f'Путь к private test котировкам (по умолчанию: {PRIVATE_TEST_CANDLES})')
    parser.add_argument('--test-news', type=str, default=None,
                       help=f'Путь к test новостям (по умолчанию: {TEST_NEWS})')
    parser.add_argument('--models-dir', type=str, default=None,
                       help=f'Директория с обученными моделями (по умолчанию: {MODELS_DIR})')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help=f'Путь к выходному файлу (по умолчанию: {PROJECT_ROOT / "submission.csv"})')
    
    args = parser.parse_args()
    
    main(
        public_test_candles=args.public_test_candles,
        private_test_candles=args.private_test_candles,
        test_news=args.test_news,
        models_dir=args.models_dir,
        output_file=args.output
    )

