"""
Загрузка и предобработка данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from src.config import (
    TRAIN_CANDLES, TRAIN_NEWS,
    PUBLIC_TEST_CANDLES, PRIVATE_TEST_CANDLES, TEST_NEWS
)
from src.ticker_extractor import TickerExtractor


class DataLoader:
    """Класс для загрузки и предобработки данных"""
    
    def __init__(self):
        self.train_candles = None
        self.train_news = None
        self.test_candles = None
        self.test_news = None
        self.all_candles = None
        self.ticker_extractor = TickerExtractor()
        
    def load_data(self, mode='train'):
        """
        Загрузка данных
        
        Args:
            mode: 'train' или 'test'
        """
        print(f"📊 Загрузка данных (режим: {mode})...")
        
        # Загружаем котировки
        if mode == 'train' or self.train_candles is None:
            self.train_candles = pd.read_csv(TRAIN_CANDLES)
            self.train_candles['begin'] = pd.to_datetime(self.train_candles['begin'])
            # Создаем ID строки для идентификации (вместо ticker)
            self.train_candles['row_id'] = range(len(self.train_candles))
            ticker_info = f", {self.train_candles['ticker'].nunique()} тикеров" if 'ticker' in self.train_candles.columns else ""
            print(f"   ✓ Train candles: {len(self.train_candles)} строк{ticker_info}")
        
        # Загружаем новости для обучения
        if mode == 'train' or self.train_news is None:
            self.train_news = pd.read_csv(TRAIN_NEWS)
            self.train_news['publish_date'] = pd.to_datetime(self.train_news['publish_date'])
            print(f"   ✓ Train news: {len(self.train_news)} новостей")
        
        # Загружаем тестовые данные
        public_test = pd.read_csv(PUBLIC_TEST_CANDLES)
        public_test['begin'] = pd.to_datetime(public_test['begin'])
        
        private_test = pd.read_csv(PRIVATE_TEST_CANDLES)
        private_test['begin'] = pd.to_datetime(private_test['begin'])
        
        self.test_candles = pd.concat([public_test, private_test], ignore_index=True)
        print(f"   ✓ Test candles: {len(self.test_candles)} строк "
              f"(public: {len(public_test)}, private: {len(private_test)})")
        
        # Загружаем тестовые новости
        self.test_news = pd.read_csv(TEST_NEWS)
        self.test_news['publish_date'] = pd.to_datetime(self.test_news['publish_date'])
        print(f"   ✓ Test news: {len(self.test_news)} новостей")
        
        # Объединяем все котировки для вычисления признаков
        self.all_candles = pd.concat([self.train_candles, self.test_candles], 
                                      ignore_index=True)
        
        # Извлекаем/определяем тикеры если их нет
        if 'ticker' not in self.all_candles.columns or self.all_candles['ticker'].isna().any():
            print("\n   ⚠️  Колонка 'ticker' отсутствует или содержит NaN")
            print("   🔍 Определяем тикеры автоматически...")
            self.all_candles = self.ticker_extractor.infer_ticker_from_candles(self.all_candles)
            # Используем inferred_ticker как ticker
            if 'inferred_ticker' in self.all_candles.columns:
                self.all_candles['ticker'] = self.all_candles['inferred_ticker']
        
        self.all_candles = self.all_candles.sort_values(['ticker', 'begin']).reset_index(drop=True)
        
        # Объединяем все новости
        self.all_news = pd.concat([self.train_news, self.test_news], 
                                   ignore_index=True)
        self.all_news = self.all_news.sort_values('publish_date').reset_index(drop=True)
        
        # Извлекаем тикеры из новостей если их нет
        if 'tickers' not in self.all_news.columns or self.all_news['tickers'].isna().all():
            self.all_news = self.ticker_extractor.extract_tickers_from_news(self.all_news)
        
        print(f"   ✓ Total candles: {len(self.all_candles)} строк")
        print(f"   ✓ Total news: {len(self.all_news)} новостей")
        
        return self
    
    def get_train_data(self) -> pd.DataFrame:
        """Получить обучающие данные"""
        return self.train_candles.copy()
    
    def get_test_data(self) -> pd.DataFrame:
        """Получить тестовые данные"""
        return self.test_candles.copy()
    
    def get_all_candles(self) -> pd.DataFrame:
        """Получить все котировки (train + test)"""
        return self.all_candles.copy()
    
    def get_all_news(self) -> pd.DataFrame:
        """Получить все новости (train + test)"""
        return self.all_news.copy()
    
    def prepare_targets_for_horizons(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовить таргеты для всех 20 горизонтов
        
        Для каждого дня t вычисляем доходность и направление для дней t+1, t+2, ..., t+20
        """
        print("\n🎯 Подготовка таргетов для всех горизонтов...")
        
        df = df.sort_values(['ticker', 'begin']).reset_index(drop=True)
        
        # Для каждого горизонта создаем таргеты
        for horizon in range(1, 21):
            # Доходность за horizon дней
            df[f'target_return_{horizon}d'] = df.groupby('ticker')['close'].transform(
                lambda x: x.shift(-horizon) / x - 1
            )
            
            # Направление (0 или 1)
            df[f'target_direction_{horizon}d'] = (
                df[f'target_return_{horizon}d'] > 0
            ).astype(int)
        
        print(f"   ✓ Создано {20} пар таргетов (return + direction)")
        
        return df
    
    def get_tickers(self) -> list:
        """Получить список всех тикеров"""
        return sorted(self.all_candles['ticker'].unique().tolist())
    
    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Получить диапазон дат"""
        return (
            self.all_candles['begin'].min(),
            self.all_candles['begin'].max()
        )

