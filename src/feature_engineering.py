"""
Feature Engineering - создание признаков из котировок и новостей
"""
import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from src.config import WINDOW_SIZE, RANDOM_SEED


class FeatureEngineer:
    """Класс для создания признаков"""
    
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.news_embeddings = None
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков из котировок
        
        Создаем технические индикаторы:
        - Моментум (изменение цены за N дней)
        - Скользящие средние
        - Волатильность
        - RSI (Relative Strength Index)
        - MACD
        - Bollinger Bands
        """
        print("\n🔧 Создание признаков из котировок...")
        
        df = df.sort_values(['ticker', 'begin']).reset_index(drop=True)
        
        # Группируем по тикерам для вычисления признаков
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            ticker_data = df[mask].copy()
            
            # 1. Базовые доходности
            ticker_data['return_1d'] = ticker_data['close'].pct_change(1)
            ticker_data['return_5d'] = ticker_data['close'].pct_change(5)
            ticker_data['return_10d'] = ticker_data['close'].pct_change(10)
            ticker_data['return_20d'] = ticker_data['close'].pct_change(20)
            
            # 2. Скользящие средние
            ticker_data['ma_5'] = ticker_data['close'].rolling(5).mean()
            ticker_data['ma_10'] = ticker_data['close'].rolling(10).mean()
            ticker_data['ma_20'] = ticker_data['close'].rolling(20).mean()
            
            # 3. Расстояние от скользящих средних
            ticker_data['dist_ma_5'] = (ticker_data['close'] - ticker_data['ma_5']) / ticker_data['ma_5']
            ticker_data['dist_ma_10'] = (ticker_data['close'] - ticker_data['ma_10']) / ticker_data['ma_10']
            ticker_data['dist_ma_20'] = (ticker_data['close'] - ticker_data['ma_20']) / ticker_data['ma_20']
            
            # 4. Волатильность (std доходностей)
            ticker_data['volatility_5'] = ticker_data['return_1d'].rolling(5).std()
            ticker_data['volatility_10'] = ticker_data['return_1d'].rolling(10).std()
            ticker_data['volatility_20'] = ticker_data['return_1d'].rolling(20).std()
            
            # 5. High-Low спред (нормализованный)
            ticker_data['hl_spread'] = (ticker_data['high'] - ticker_data['low']) / ticker_data['close']
            ticker_data['hl_spread_ma5'] = ticker_data['hl_spread'].rolling(5).mean()
            
            # 6. Объем торгов (логарифм и нормализация)
            ticker_data['volume_log'] = np.log1p(ticker_data['volume'])
            ticker_data['volume_ma5'] = ticker_data['volume'].rolling(5).mean()
            ticker_data['volume_ratio'] = ticker_data['volume'] / (ticker_data['volume_ma5'] + 1e-8)
            
            # 7. RSI (Relative Strength Index)
            ticker_data['rsi_14'] = self._calculate_rsi(ticker_data['close'], period=14)
            
            # 8. MACD
            ema_12 = ticker_data['close'].ewm(span=12, adjust=False).mean()
            ema_26 = ticker_data['close'].ewm(span=26, adjust=False).mean()
            ticker_data['macd'] = ema_12 - ema_26
            ticker_data['macd_signal'] = ticker_data['macd'].ewm(span=9, adjust=False).mean()
            ticker_data['macd_diff'] = ticker_data['macd'] - ticker_data['macd_signal']
            
            # 9. Bollinger Bands
            ticker_data['bb_middle'] = ticker_data['close'].rolling(20).mean()
            bb_std = ticker_data['close'].rolling(20).std()
            ticker_data['bb_upper'] = ticker_data['bb_middle'] + 2 * bb_std
            ticker_data['bb_lower'] = ticker_data['bb_middle'] - 2 * bb_std
            ticker_data['bb_position'] = (ticker_data['close'] - ticker_data['bb_lower']) / (
                ticker_data['bb_upper'] - ticker_data['bb_lower'] + 1e-8
            )
            
            # 10. Тренд (линейная регрессия slope)
            ticker_data['trend_5'] = ticker_data['close'].rolling(5).apply(
                lambda x: self._calculate_slope(x), raw=True
            )
            ticker_data['trend_10'] = ticker_data['close'].rolling(10).apply(
                lambda x: self._calculate_slope(x), raw=True
            )
            
            # Обновляем данные в основном DataFrame
            for col in ticker_data.columns:
                if col not in df.columns or col.startswith(('return_', 'ma_', 'dist_', 'volatility_',
                                                             'hl_', 'volume_', 'rsi_', 'macd', 'bb_',
                                                             'trend_')):
                    df.loc[mask, col] = ticker_data[col].values
        
        print(f"   ✓ Создано {len([c for c in df.columns if any(c.startswith(p) for p in ['return_', 'ma_', 'dist_', 'volatility_', 'hl_', 'volume_', 'rsi_', 'macd', 'bb_', 'trend_'])])} признаков из котировок")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Вычисление RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_slope(self, y: np.ndarray) -> float:
        """Вычисление наклона линейной регрессии"""
        if len(y) < 2 or np.isnan(y).any():
            return 0
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def create_news_features(self, candles_df: pd.DataFrame, news_df: pd.DataFrame, 
                            use_embeddings: bool = False) -> pd.DataFrame:
        """
        Создание признаков из новостей
        
        Для каждого дня и тикера агрегируем новости за последние N дней
        
        Args:
            candles_df: DataFrame с котировками
            news_df: DataFrame с новостями  
            use_embeddings: Использовать ли эмбеддинги (требует больше времени)
        """
        print("\n📰 Создание признаков из новостей...")
        
        df = candles_df.copy()
        
        # Базовые статистики новостей
        df['news_count_1d'] = 0
        df['news_count_7d'] = 0
        df['news_count_30d'] = 0
        df['news_sentiment_score'] = 0.5  # Нейтральный по умолчанию
        df['news_title_length_mean'] = 0
        df['news_has_ticker'] = 0
        
        # Для каждой строки в candles находим соответствующие новости
        for idx, row in df.iterrows():
            ticker = row['ticker']
            date = row['begin']
            
            # Новости за последний день (с учетом задержки в 1 день)
            news_1d = news_df[
                (news_df['publish_date'] >= date - pd.Timedelta(days=2)) &
                (news_df['publish_date'] < date) &
                (news_df['tickers'].str.contains(ticker, na=False, regex=False) |
                 news_df['title'].str.contains(ticker, na=False, case=False) |
                 news_df['publication'].str.contains(ticker, na=False, case=False))
            ]
            
            # Новости за последние 7 дней
            news_7d = news_df[
                (news_df['publish_date'] >= date - pd.Timedelta(days=8)) &
                (news_df['publish_date'] < date) &
                (news_df['tickers'].str.contains(ticker, na=False, regex=False) |
                 news_df['title'].str.contains(ticker, na=False, case=False) |
                 news_df['publication'].str.contains(ticker, na=False, case=False))
            ]
            
            # Количество новостей
            df.at[idx, 'news_count_1d'] = len(news_1d)
            df.at[idx, 'news_count_7d'] = len(news_7d)
            
            # Средняя длина заголовков
            if len(news_7d) > 0:
                df.at[idx, 'news_title_length_mean'] = news_7d['title'].str.len().mean()
                df.at[idx, 'news_has_ticker'] = 1
                
                # Простой sentiment score на основе ключевых слов
                sentiment_score = self._simple_sentiment(news_7d)
                df.at[idx, 'news_sentiment_score'] = sentiment_score
            
            if idx % 1000 == 0:
                print(f"   Обработано {idx}/{len(df)} строк...", end='\r')
        
        print(f"\n   ✓ Создано {len([c for c in df.columns if c.startswith('news_')])} признаков из новостей")
        
        return df
    
    def _simple_sentiment(self, news_df: pd.DataFrame) -> float:
        """
        Простой sentiment analysis на основе ключевых слов
        
        Returns:
            float: sentiment score от 0 (негативный) до 1 (позитивный)
        """
        if len(news_df) == 0:
            return 0.5
        
        # Ключевые слова
        positive_words = ['рост', 'прибыль', 'успех', 'увеличение', 'развитие', 
                         'положительн', 'выгодн', 'улучшение', 'повышение']
        negative_words = ['убыт', 'снижение', 'падение', 'кризис', 'риск',
                         'отрицательн', 'проблем', 'ухудшение', 'потер']
        
        sentiment_scores = []
        
        for _, row in news_df.iterrows():
            text = (str(row['title']) + ' ' + str(row['publication'])).lower()
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count + neg_count > 0:
                score = pos_count / (pos_count + neg_count)
            else:
                score = 0.5
            
            sentiment_scores.append(score)
        
        return np.mean(sentiment_scores) if sentiment_scores else 0.5
    
    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Заполнение пропущенных значений"""
        print("\n🔄 Заполнение пропущенных значений...")
        
        # Числовые колонки
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not col.startswith('target_')]
        
        # Заполняем NaN медианными значениями для каждого тикера
        for ticker in df['ticker'].unique():
            mask = df['ticker'] == ticker
            df.loc[mask, feature_cols] = df.loc[mask, feature_cols].fillna(
                df.loc[mask, feature_cols].median()
            )
        
        # Оставшиеся NaN заполняем глобальной медианой
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        # Финальное заполнение нулями (если остались NaN)
        df[feature_cols] = df[feature_cols].fillna(0)
        
        print(f"   ✓ Пропущенные значения заполнены")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Получить список колонок с признаками (исключая target и служебные)"""
        exclude_prefixes = ['target_', 'Unnamed']
        exclude_cols = ['ticker', 'begin', 'open', 'close', 'high', 'low', 'volume']
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and not any(col.startswith(prefix) for prefix in exclude_prefixes)]
        
        return feature_cols

