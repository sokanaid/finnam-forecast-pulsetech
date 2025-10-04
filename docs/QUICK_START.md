# ⚡ Быстрый старт FORECAST

## 3 команды для запуска

```bash
# 1. Установка зависимостей
pip install pandas numpy scikit-learn lightgbm joblib

# 2. Обучение модели (15-30 минут)
python train.py

# 3. Создание предсказаний (1-5 минут)
python predict.py
```

**Результат:** файл `submission.csv` готов для отправки на лидерборд!

---

## 🎯 Что делает решение?

Прогнозирует **вероятности роста цен акций** на 1, 2, ..., 20 торговых дней вперед, используя:
- Исторические котировки (OHLCV)
- Технические индикаторы (RSI, MACD, Bollinger Bands)
- Новости о компаниях
- Ансамбль из 20 моделей LightGBM

---

## 📂 Требуемые файлы данных

Разместите файлы в `data/raw/`:
- `train_candles.csv` - обучающие котировки
- `train_news.csv` - обучающие новости
- `public_test_candles.csv` - публичный тест
- `private_test_candles.csv` - приватный тест  
- `test_news.csv` - тестовые новости

---

## 🔧 Запуск с параметрами

### Обучение на своих данных
```bash
python train.py \
  --train-candles path/to/train_candles.csv \
  --train-news path/to/train_news.csv \
  --output-dir my_models/
```

### Предсказание с указанием путей
```bash
python predict.py \
  --public-test-candles path/to/public_test.csv \
  --private-test-candles path/to/private_test.csv \
  --test-news path/to/test_news.csv \
  --models-dir my_models/ \
  --output my_submission.csv
```

---

## 📖 Документация

- **README.md** - полное описание проекта
- **USAGE.md** - подробные примеры использования
- **SUMMARY.md** - итоговая сводка решения
- **QUICK_START.md** - этот файл

---

## 💡 Справка

```bash
python train.py --help      # Справка по обучению
python predict.py --help    # Справка по предсказанию
```

---

## ⚠️ Важно: Извлечение тикеров

Если в данных отсутствует колонка `ticker`, решение автоматически:
1. **Извлекает тикеры из новостей** (Сбербанк → SBER, Газпром → GAZP)
2. **Кластеризует котировки** и присваивает тикеры на основе цен и объемов

Поддерживается 35+ тикеров: SBER, GAZP, LKOH, YNDX, VTBR, TCSG, MGNT и др.

---

## 🚀 Готово к работе!

Просто запустите 3 команды выше, и у вас будет готовый `submission.csv` для хакатона!

