"""
Модель прогнозирования вероятностей роста
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

from src.config import RANDOM_SEED, N_HORIZONS, MODELS_DIR

# Пробуем импортировать LightGBM, если не получается - используем sklearn
try:
    from lightgbm import LGBMClassifier
    USE_LIGHTGBM = True
except (ImportError, OSError):
    USE_LIGHTGBM = False
    print("⚠️  LightGBM недоступен, используем sklearn GradientBoostingClassifier")


class ForecastModel:
    """Модель для прогнозирования вероятностей роста на разных горизонтах"""
    
    def __init__(self, n_horizons=N_HORIZONS):
        self.n_horizons = n_horizons
        self.models = {}  # Словарь моделей для каждого горизонта
        self.feature_columns = []
        
    def train(self, df: pd.DataFrame, feature_cols: List[str], 
             val_size: float = 0.2, verbose: bool = True):
        """
        Обучение моделей для всех горизонтов
        
        Args:
            df: DataFrame с признаками и таргетами
            feature_cols: Список колонок-признаков
            val_size: Размер валидационной выборки
            verbose: Выводить ли информацию об обучении
        """
        print("\n🎓 Обучение моделей для всех горизонтов...")
        print(f"   Количество горизонтов: {self.n_horizons}")
        print(f"   Количество признаков: {len(feature_cols)}")
        print(f"   Размер обучающей выборки: {len(df)}")
        
        self.feature_columns = feature_cols
        
        # Удаляем строки с NaN в таргетах
        train_df = df.copy()
        
        # Обучаем модель для каждого горизонта
        for horizon in range(1, self.n_horizons + 1):
            if verbose:
                print(f"\n   📊 Горизонт {horizon} дней:")
            
            # Таргет для этого горизонта
            target_col = f'target_direction_{horizon}d'
            
            if target_col not in train_df.columns:
                print(f"      ⚠️  Таргет {target_col} не найден, пропускаем")
                continue
            
            # Удаляем строки с NaN в таргете для этого горизонта
            valid_mask = train_df[target_col].notna()
            X = train_df.loc[valid_mask, feature_cols].values
            y = train_df.loc[valid_mask, target_col].values
            
            if len(y) < 100:
                print(f"      ⚠️  Недостаточно данных ({len(y)} строк), пропускаем")
                continue
            
            # Разбиваем на train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_size, random_state=RANDOM_SEED, stratify=y
            )
            
            # Создаем и обучаем модель
            if USE_LIGHTGBM:
                model = LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbose=-1,
                    class_weight='balanced'
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    verbose=False
                )
            else:
                # Используем sklearn GradientBoosting если LightGBM недоступен
                model = GradientBoostingClassifier(
                    n_estimators=100,  # Меньше для скорости
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=RANDOM_SEED,
                    verbose=0
                )
                
                model.fit(X_train, y_train)
            
            # Оценка на валидации
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            auc = roc_auc_score(y_val, y_pred_proba)
            acc = accuracy_score(y_val, y_pred)
            
            if verbose:
                print(f"      Train size: {len(X_train)}, Val size: {len(X_val)}")
                print(f"      Positive rate: {y_train.mean():.3f}")
                print(f"      Val AUC: {auc:.4f}, Val Accuracy: {acc:.4f}")
            
            # Сохраняем модель
            self.models[horizon] = model
        
        print(f"\n   ✅ Обучено {len(self.models)} моделей")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Предсказание вероятностей роста для всех горизонтов
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            DataFrame с колонками ticker, begin, p1, p2, ..., p20
        """
        print("\n🔮 Предсказание вероятностей роста...")
        
        if not self.models:
            raise ValueError("Модели не обучены! Сначала вызовите train()")
        
        if not self.feature_columns:
            raise ValueError("Список признаков не определен!")
        
        # Проверяем наличие всех признаков
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            print(f"      ⚠️  Отсутствующие признаки: {missing_features}")
            # Добавляем недостающие признаки с нулями
            for feat in missing_features:
                df[feat] = 0
        
        X = df[self.feature_columns].values
        
        # Предсказываем для каждого горизонта
        predictions = pd.DataFrame()
        predictions['ticker'] = df['ticker'].values
        predictions['begin'] = df['begin'].values
        
        for horizon in range(1, self.n_horizons + 1):
            if horizon in self.models:
                model = self.models[horizon]
                proba = model.predict_proba(X)[:, 1]
                
                # Клиппинг вероятностей в разумный диапазон
                proba = np.clip(proba, 0.05, 0.95)
            else:
                # Если модели нет, используем нейтральное значение
                proba = np.full(len(df), 0.5)
            
            predictions[f'p{horizon}'] = proba
        
        print(f"   ✓ Создано {len(predictions)} предсказаний для {self.n_horizons} горизонтов")
        print(f"\n   📊 Статистика вероятностей:")
        prob_cols = [f'p{i}' for i in range(1, self.n_horizons + 1)]
        print(f"      Средняя: {predictions[prob_cols].mean().mean():.4f}")
        print(f"      Std: {predictions[prob_cols].std().mean():.4f}")
        
        return predictions
    
    def save(self, path: Path = None):
        """Сохранение моделей"""
        if path is None:
            path = MODELS_DIR / "forecast_models.pkl"
        
        print(f"\n💾 Сохранение моделей в {path}...")
        
        model_data = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'n_horizons': self.n_horizons
        }
        
        joblib.dump(model_data, path)
        print(f"   ✓ Сохранено {len(self.models)} моделей")
        
    def load(self, path: Path = None):
        """Загрузка моделей"""
        if path is None:
            path = MODELS_DIR / "forecast_models.pkl"
        
        print(f"\n📂 Загрузка моделей из {path}...")
        
        model_data = joblib.load(path)
        self.models = model_data['models']
        self.feature_columns = model_data['feature_columns']
        self.n_horizons = model_data['n_horizons']
        
        print(f"   ✓ Загружено {len(self.models)} моделей")
        print(f"   ✓ Количество признаков: {len(self.feature_columns)}")
        
        return self
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[int, pd.DataFrame]:
        """
        Получить важность признаков для каждого горизонта
        
        Args:
            top_n: Количество топ признаков для каждого горизонта
            
        Returns:
            Словарь {horizon: DataFrame with feature importance}
        """
        importance_dict = {}
        
        for horizon, model in self.models.items():
            try:
                importance = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                })
                importance = importance.sort_values('importance', ascending=False).head(top_n)
                importance_dict[horizon] = importance
            except AttributeError:
                # Если модель не поддерживает feature_importances_
                print(f"   ⚠️  Модель для горизонта {horizon} не поддерживает feature_importances_")
        
        return importance_dict

