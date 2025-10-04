"""
Скрипт для оценки модели FORECAST
"""
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, p_up: np.ndarray) -> dict:
    """
    Расчет метрик для одного горизонта
    
    Args:
        y_true: Истинная доходность
        y_pred: Предсказанная доходность
        p_up: Предсказанная вероятность роста
        
    Returns:
        dict с метриками: MAE, Brier, DA
    """
    # MAE - Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Brier Score - для вероятности роста
    y_up_true = (y_true > 0).astype(int)  # 1 если рост, 0 если падение
    brier = np.mean((y_up_true - p_up) ** 2)
    
    # DA - Direction Accuracy
    y_up_pred = (p_up > 0.5).astype(int)
    da = np.mean((np.sign(y_true) == np.sign(y_up_pred - 0.5)))
    
    return {
        'MAE': mae,
        'Brier': brier,
        'DA': da
    }


def calculate_score(mae: float, brier: float, da: float,
                   mae_baseline: float = None, 
                   brier_baseline: float = None) -> float:
    """
    Расчет итоговой метрики Score
    
    Score = 0.7 * MAE_norm + 0.3 * Brier_norm + 0.1 * DA
    
    где MAE_norm = 1 - MAE/MAE_baseline
        Brier_norm = 1 - Brier/Brier_baseline
    """
    if mae_baseline is not None and brier_baseline is not None:
        mae_norm = 1 - mae / mae_baseline
        brier_norm = 1 - brier / brier_baseline
    else:
        # Если нет baseline, используем ненормализованные метрики
        # Нормализуем так, чтобы Score был от 0 до 1
        mae_norm = 1 - min(mae, 1.0)  # MAE не должен быть > 1
        brier_norm = 1 - min(brier, 1.0)  # Brier не должен быть > 1
    
    score = 0.7 * mae_norm + 0.3 * brier_norm + 0.1 * da
    
    return score


def evaluate_submission(submission_path: str, test_data_path: str = None):
    """
    Оценка submission файла
    
    Args:
        submission_path: Путь к файлу с предсказаниями
        test_data_path: Путь к файлу с истинными значениями (если есть)
    """
    print("=" * 70)
    print("📊 ОЦЕНКА SUBMISSION")
    print("=" * 70)
    
    # Загружаем submission
    print(f"\n📂 Загрузка submission: {submission_path}")
    submission = pd.read_csv(submission_path)
    
    print(f"   ✓ Строк: {len(submission)}")
    print(f"   ✓ Колонок: {len(submission.columns)}")
    print(f"   ✓ Тикеров: {submission['ticker'].nunique()}")
    
    # Проверяем формат
    required_cols = ['ticker'] + [f'p{i}' for i in range(1, 21)]
    missing_cols = set(required_cols) - set(submission.columns)
    if missing_cols:
        print(f"\n   ⚠️  Отсутствуют колонки: {missing_cols}")
        return
    
    print(f"\n   ✅ Формат корректный")
    
    # Статистика вероятностей
    prob_cols = [f'p{i}' for i in range(1, 21)]
    
    print(f"\n📊 Статистика вероятностей:")
    for col in prob_cols[:5]:  # Показываем первые 5 горизонтов
        print(f"   {col}: mean={submission[col].mean():.4f}, "
              f"std={submission[col].std():.4f}, "
              f"min={submission[col].min():.4f}, "
              f"max={submission[col].max():.4f}")
    
    print(f"   ... (и еще {len(prob_cols)-5} горизонтов)")
    
    # Проверяем, что вероятности в диапазоне [0, 1]
    for col in prob_cols:
        if submission[col].min() < 0 or submission[col].max() > 1:
            print(f"\n   ⚠️  Вероятности в {col} вне диапазона [0, 1]!")
    
    # Если есть test данные с таргетами, считаем метрики
    if test_data_path and Path(test_data_path).exists():
        print(f"\n📂 Загрузка test данных: {test_data_path}")
        test_data = pd.read_csv(test_data_path)
        test_data['begin'] = pd.to_datetime(test_data['begin'])
        
        # TODO: Реализовать расчет метрик при наличии таргетов
        print(f"   ⚠️  Расчет метрик по test данным пока не реализован")
    else:
        print(f"\n   ℹ️  Test данные с таргетами не предоставлены")
    
    print("\n" + "=" * 70)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА")
    print("=" * 70)


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка submission файла')
    parser.add_argument('submission', type=str, help='Путь к submission файлу')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Путь к test данным с таргетами (опционально)')
    
    args = parser.parse_args()
    
    evaluate_submission(args.submission, args.test_data)


if __name__ == "__main__":
    main()

