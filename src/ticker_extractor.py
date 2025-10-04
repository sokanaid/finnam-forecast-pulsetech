"""
Извлечение тикеров из данных с помощью регулярных выражений
"""
import pandas as pd
import numpy as np
import re
from typing import List, Optional


class TickerExtractor:
    """Класс для извлечения и определения тикеров"""
    
    # Список известных российских тикеров
    KNOWN_TICKERS = [
        'SBER', 'GAZP', 'LKOH', 'GMKN', 'NVTK', 'ROSN', 'TATN', 
        'MGNT', 'MTSS', 'ALRS', 'SNGS', 'NLMK', 'CHMF', 'VTBR',
        'YNDX', 'PLZL', 'AFLT', 'MOEX', 'RTKM', 'FEES', 'HYDR',
        'IRAO', 'PHOR', 'RUAL', 'TCSG', 'AFKS', 'PIKK', 'POLY',
        'MAIL', 'OZON', 'FIXP', 'CBOM', 'TRNFP', 'UPRO', 'MAGN'
    ]
    
    # Соответствие компаний тикерам (для поиска в тексте)
    COMPANY_TO_TICKER = {
        'сбер': 'SBER', 'сбербанк': 'SBER',
        'газпром': 'GAZP',
        'лукойл': 'LKOH', 'lukoil': 'LKOH',
        'норникель': 'GMKN', 'норильский никель': 'GMKN',
        'новатэк': 'NVTK', 'novatek': 'NVTK',
        'роснефть': 'ROSN',
        'татнефть': 'TATN',
        'магнит': 'MGNT',
        'мтс': 'MTSS',
        'алроса': 'ALRS',
        'сургутнефтегаз': 'SNGS',
        'нлмк': 'NLMK', 'новолипецк': 'NLMK',
        'северсталь': 'CHMF',
        'втб': 'VTBR',
        'яндекс': 'YNDX', 'yandex': 'YNDX',
        'полюс': 'PLZL',
        'аэрофлот': 'AFLT',
        'мосбиржа': 'MOEX', 'московская биржа': 'MOEX',
        'ростелеком': 'RTKM',
        'фск': 'FEES', 'фск еэс': 'FEES',
        'русгидро': 'HYDR',
        'интер рао': 'IRAO',
        'фосагро': 'PHOR',
        'русал': 'RUAL',
        'тинькофф': 'TCSG', 'тинькоф': 'TCSG', 'тиньков': 'TCSG', 'tcs': 'TCSG',
        'афк система': 'AFKS', 'система': 'AFKS',
        'пик': 'PIKK',
        'полиметалл': 'POLY',
        'mail': 'MAIL', 'мэйл': 'MAIL',
        'озон': 'OZON', 'ozon': 'OZON',
        'fix price': 'FIXP', 'фикс прайс': 'FIXP',
        'совкомбанк': 'CBOM',
        'транснефть': 'TRNFP',
        'юнипро': 'UPRO',
        'ммк': 'MAGN', 'магнитогорск': 'MAGN'
    }
    
    def __init__(self):
        # Создаем регулярные выражения для поиска
        self.ticker_pattern = re.compile(r'\b(' + '|'.join(self.KNOWN_TICKERS) + r')\b', re.IGNORECASE)
        
    def extract_tickers_from_text(self, text: str) -> List[str]:
        """
        Извлечь тикеры из текста
        
        Args:
            text: Текст для поиска
            
        Returns:
            Список найденных тикеров
        """
        if pd.isna(text) or not text:
            return []
        
        text = str(text).lower()
        tickers = set()
        
        # 1. Ищем прямые упоминания тикеров (например, "SBER")
        direct_matches = self.ticker_pattern.findall(text)
        tickers.update([t.upper() for t in direct_matches])
        
        # 2. Ищем упоминания компаний
        for company, ticker in self.COMPANY_TO_TICKER.items():
            if company in text:
                tickers.add(ticker)
        
        return list(tickers)
    
    def extract_tickers_from_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечь тикеры из новостей
        
        Args:
            news_df: DataFrame с новостями
            
        Returns:
            DataFrame с добавленной колонкой 'tickers'
        """
        print("🔍 Извлечение тикеров из новостей...")
        
        news_df = news_df.copy()
        
        # Извлекаем тикеры из title и publication
        news_df['tickers'] = news_df.apply(
            lambda row: self.extract_tickers_from_text(
                str(row.get('title', '')) + ' ' + str(row.get('publication', ''))
            ), axis=1
        )
        
        # Преобразуем список в строку через запятую
        news_df['tickers_str'] = news_df['tickers'].apply(
            lambda x: ','.join(x) if x else ''
        )
        
        found_count = (news_df['tickers'].str.len() > 0).sum()
        print(f"   ✓ Найдено тикеров в {found_count}/{len(news_df)} новостях")
        
        return news_df
    
    def infer_ticker_from_candles(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Определить тикер для каждой строки котировок на основе паттернов
        
        Используем кластеризацию по:
        - Диапазону цен (price range)
        - Средней цене
        - Волатильности
        - Объему торгов
        
        Args:
            candles_df: DataFrame с котировками
            
        Returns:
            DataFrame с добавленной колонкой 'inferred_ticker'
        """
        print("\n🔍 Определение тикеров из котировок...")
        
        df = candles_df.copy()
        
        # Если тикер уже есть, используем его
        if 'ticker' in df.columns:
            df['inferred_ticker'] = df['ticker']
            print(f"   ✓ Используем существующую колонку 'ticker'")
            return df
        
        # Сортируем по дате
        df = df.sort_values('begin').reset_index(drop=True)
        
        # Признаки для кластеризации
        df['price_range'] = df['high'] - df['low']
        df['avg_price'] = (df['high'] + df['low']) / 2
        df['price_volatility'] = df['price_range'] / df['avg_price']
        df['log_volume'] = np.log1p(df['volume'])
        
        # Используем rolling mean для сглаживания
        window = 5
        df['avg_price_smooth'] = df['avg_price'].rolling(window, min_periods=1).mean()
        df['log_volume_smooth'] = df['log_volume'].rolling(window, min_periods=1).mean()
        
        # Простая кластеризация на основе ценовых уровней
        # Предполагаем, что разные тикеры торгуются в разных ценовых диапазонах
        from sklearn.cluster import KMeans
        
        # Нормализуем признаки
        features = df[['avg_price_smooth', 'log_volume_smooth', 'price_volatility']].fillna(0)
        
        # Определяем количество кластеров (примерно 10-15 тикеров)
        n_clusters = min(len(self.KNOWN_TICKERS), len(df) // 100)
        n_clusters = max(n_clusters, 10)  # Минимум 10 кластеров
        
        if len(df) > n_clusters * 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(features)
            
            # Присваиваем тикеры кластерам (сортируем по средней цене)
            cluster_prices = df.groupby('cluster')['avg_price'].mean().sort_values()
            cluster_to_ticker = {
                cluster: self.KNOWN_TICKERS[i % len(self.KNOWN_TICKERS)]
                for i, cluster in enumerate(cluster_prices.index)
            }
            
            df['inferred_ticker'] = df['cluster'].map(cluster_to_ticker)
        else:
            # Если данных мало, просто нумеруем группы
            df['inferred_ticker'] = 'UNKNOWN'
        
        print(f"   ✓ Определено {df['inferred_ticker'].nunique()} уникальных тикеров")
        print(f"   Тикеры: {sorted(df['inferred_ticker'].unique())[:10]}...")
        
        return df
    
    def assign_tickers_by_sequence(self, candles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Более простой метод: присваиваем тикеры на основе последовательности
        
        Предполагаем, что данные отсортированы и идут блоками по тикерам
        """
        print("\n🔍 Присвоение тикеров по последовательности...")
        
        df = candles_df.copy()
        
        # Если тикер уже есть, используем его
        if 'ticker' in df.columns:
            df['inferred_ticker'] = df['ticker']
            print(f"   ✓ Используем существующую колонку 'ticker'")
            return df
        
        # Сортируем по дате
        df = df.sort_values('begin').reset_index(drop=True)
        
        # Детектируем смену паттерна (переход к другому тикеру)
        # Используем изменения в ценовом уровне
        df['price_level'] = df['close'].rolling(10, min_periods=1).mean()
        df['price_change'] = df['price_level'].pct_change().abs()
        
        # Большие скачки цены могут означать смену тикера
        threshold = df['price_change'].quantile(0.95)
        df['ticker_break'] = (df['price_change'] > threshold).astype(int)
        
        # Создаем группы
        df['ticker_group'] = df['ticker_break'].cumsum()
        
        # Присваиваем тикеры группам
        unique_groups = df['ticker_group'].unique()
        group_to_ticker = {
            group: self.KNOWN_TICKERS[i % len(self.KNOWN_TICKERS)]
            for i, group in enumerate(sorted(unique_groups))
        }
        
        df['inferred_ticker'] = df['ticker_group'].map(group_to_ticker)
        
        print(f"   ✓ Определено {df['inferred_ticker'].nunique()} уникальных тикеров")
        print(f"   Тикеры: {sorted(df['inferred_ticker'].unique())}")
        
        return df

