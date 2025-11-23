import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import time
import threading
import random

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuturesIndicatorBot:
    def __init__(self):
        self.base_url = "https://api.bybit.com"
        
        # === НАСТРОЙКИ TELEGRAM ===
        self.telegram_token = "8542640031:AAH5yPkJiDCFoHSo8IwuTRY4HodYLj34P54"
        self.telegram_chat_id = "352030600"
        # =========================
        
        self.last_update_id = None
        self._colab_keep_alive_thread = None

    def _start_colab_keep_alive(self):
        """Безопасная фоновая задача для Colab"""
        def keep_alive():
            while True:
                # Минимальное вычисление для активности
                _ = 1 + 1
                
                # Случайный интервал от 6 до 10 часов
                sleep_time = random.randint(21600, 36000)
                time.sleep(sleep_time)
        
        self._colab_keep_alive_thread = threading.Thread(target=keep_alive)
        self._colab_keep_alive_thread.daemon = True
        self._colab_keep_alive_thread.start()

    def send_telegram_message(self, message):
        """Отправка сообщения в Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ошибка отправки в Telegram: {e}")
            return False

    def get_klines(self, symbol, interval, limit=200):
        """Получение исторических данных с большим лимитом для точности"""
        try:
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['retCode'] != 0:
                logger.error(f"Ошибка API: {data['retMsg']}")
                return None
                
            klines = data['result']['list']
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol}: {e}")
            return None

    def calculate_rsi(self, close, period):
        """Точный расчет RSI с EMA сглаживанием"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Используем EMA для сглаживания как в большинстве платформ
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50

    def calculate_rsi_multiple(self, close):
        """Расчет RSI для трех периодов"""
        return {
            'RSI6': self.calculate_rsi(close, 6),
            'RSI12': self.calculate_rsi(close, 12),
            'RSI24': self.calculate_rsi(close, 24)
        }

    def calculate_stoch_rsi_correct(self, close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
        """ТОЧНЫЙ расчет Stochastic RSI как в приложении"""
        # Сначала вычисляем RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi_series = 100 - (100 / (1 + rs))
        
        # Stochastic от RSI
        lowest_rsi = rsi_series.rolling(stoch_period).min()
        highest_rsi = rsi_series.rolling(stoch_period).max()
        
        # Избегаем деления на ноль
        stoch_rsi = 100 * (rsi_series - lowest_rsi) / (highest_rsi - lowest_rsi)
        stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], 50).fillna(50)
        
        # Сглаживание K и D линий
        k_line = stoch_rsi.rolling(k_smooth).mean()
        d_line = k_line.rolling(d_smooth).mean()
        
        return {
            'K': k_line.iloc[-1] if not k_line.empty and not pd.isna(k_line.iloc[-1]) else 50,
            'D': d_line.iloc[-1] if not d_line.empty and not pd.isna(d_line.iloc[-1]) else 50
        }

    def calculate_williams_r_correct(self, high, low, close):
        """Точный расчет Williams %R как в приложении"""
        period_14 = 14
        period_20 = 20
        
        # Для периода 14
        highest_14 = high.rolling(period_14).max()
        lowest_14 = low.rolling(period_14).min()
        wr_14 = -100 * (highest_14 - close) / (highest_14 - lowest_14)
        
        # Для периода 20  
        highest_20 = high.rolling(period_20).max()
        lowest_20 = low.rolling(period_20).min()
        wr_20 = -100 * (highest_20 - close) / (highest_20 - lowest_20)
        
        # Обработка крайних случаев
        wr_14 = wr_14.replace([np.inf, -np.inf], -50).fillna(-50)
        wr_20 = wr_20.replace([np.inf, -np.inf], -50).fillna(-50)
        
        return {
            'WR14': wr_14.iloc[-1] if not wr_14.empty and not pd.isna(wr_14.iloc[-1]) else -50,
            'WR20': wr_20.iloc[-1] if not wr_20.empty and not pd.isna(wr_20.iloc[-1]) else -50
        }

    def calculate_kdj_correct(self, high, low, close, period=9, d_period=3):
        """ТОЧНЫЙ расчет KDJ как в приложении биржи"""
        # Расчет RSV (Raw Stochastic Value)
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        
        rsv = 100 * (close - lowest_low) / (highest_high - lowest_low)
        rsv = rsv.replace([np.inf, -np.inf], 50).fillna(50)
        
        # Инициализация K и D линий
        K = pd.Series(index=rsv.index, dtype=float)
        D = pd.Series(index=rsv.index, dtype=float)
        
        # Первые значения
        if not rsv.empty:
            K.iloc[0] = 50  # Начальное значение K
            D.iloc[0] = 50  # Начальное значение D
            
            # Расчет K и D по стандартной формуле KDJ (рекурсивное сглаживание)
            for i in range(1, len(rsv)):
                K.iloc[i] = (2/3) * K.iloc[i-1] + (1/3) * rsv.iloc[i]
                D.iloc[i] = (2/3) * D.iloc[i-1] + (1/3) * K.iloc[i]
        else:
            K = pd.Series([50])
            D = pd.Series([50])
        
        J = 3 * K - 2 * D
        
        return {
            'K': K.iloc[-1] if not K.empty and not pd.isna(K.iloc[-1]) else 50,
            'D': D.iloc[-1] if not D.empty and not pd.isna(D.iloc[-1]) else 50,
            'J': J.iloc[-1] if not J.empty and not pd.isna(J.iloc[-1]) else 50
        }

    def calculate_macd(self, close, fast=12, slow=26, signal=9):
        """Расчет MACD как в приложении"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd_histogram = dif - dea
        
        return {
            'DIF': dif.iloc[-1] if not dif.empty and not pd.isna(dif.iloc[-1]) else 0,
            'DEA': dea.iloc[-1] if not dea.empty and not pd.isna(dea.iloc[-1]) else 0,
            'MACD': macd_histogram.iloc[-1] if not macd_histogram.empty and not pd.isna(macd_histogram.iloc[-1]) else 0
        }

    def calculate_volume_ma(self, volume):
        """Расчет скользящих средних объема"""
        ma5 = volume.rolling(5).mean()
        ma10 = volume.rolling(10).mean()
        
        return {
            'VOLUME': volume.iloc[-1] if not volume.empty else 0,
            'MA5': ma5.iloc[-1] if not ma5.empty and not pd.isna(ma5.iloc[-1]) else volume.iloc[-1],
            'MA10': ma10.iloc[-1] if not ma10.empty and not pd.isna(ma10.iloc[-1]) else volume.iloc[-1]
        }

    def get_indicators_for_timeframe(self, symbol, interval):
        """Получение всех индикаторов для указанного таймфрейма"""
        df = self.get_klines(symbol, interval, limit=200)
        if df is None or df.empty:
            return None
            
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # Рассчитываем все индикаторы как в приложении
        indicators = {
            'RSI': self.calculate_rsi_multiple(close),
            'StochRSI': self.calculate_stoch_rsi_correct(close),
            'WilliamsR': self.calculate_williams_r_correct(high, low, close),
            'KDJ': self.calculate_kdj_correct(high, low, close),
            'MACD': self.calculate_macd(close),
            'VolumeMA': self.calculate_volume_ma(volume),
            'current_price': close.iloc[-1],
            'price_change': ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0
        }
        
        return indicators

    def format_indicators_report(self, symbol, indicators_1d, indicators_4h, indicators_1h, indicators_5m):
        """Форматирование отчета с индикаторами как в приложении"""
        if indicators_1d is None or indicators_4h is None or indicators_1h is None or indicators_5m is None:
            return f"❌ Не удалось получить данные для {symbol}"
        
        report = f"📊 <b>АНАЛИЗ {symbol}</b>\n"
        report += f"💰 Текущая цена: <b>{indicators_1d['current_price']:.6f}</b>\n"
        report += f"📈 Изменение: <b>{indicators_1d['price_change']:+.2f}%</b>\n\n"
        
        # БЛОК 1: Дневной таймфрейм
        report += "📅 <b>ДНЕВНОЙ ТАЙМФРЕЙМ (1D)</b>\n"
        
        # Stoch RSI
        stoch_1d = indicators_1d['StochRSI']
        report += f"Stoch RSI: <b>{stoch_1d['K']:.2f}</b>  <b>{stoch_1d['D']:.2f}</b>\n"
        
        # RSI multiple
        rsi_1d = indicators_1d['RSI']
        report += f"RSI6: <b>{rsi_1d['RSI6']:.2f}</b>  "
        report += f"RSI12: <b>{rsi_1d['RSI12']:.2f}</b>  "
        report += f"RSI24: <b>{rsi_1d['RSI24']:.2f}</b>\n"
        
        # Williams %R
        wr_1d = indicators_1d['WilliamsR']
        report += f"WR14: <b>{wr_1d['WR14']:.5f}</b>  "
        report += f"WR20: <b>{wr_1d['WR20']:.5f}</b>\n"
        
        # KDJ
        kdj_1d = indicators_1d['KDJ']
        report += f"KDJ(9,3,3): <b>{kdj_1d['K']:.3f}</b>  "
        report += f"D: <b>{kdj_1d['D']:.3f}</b>  "
        report += f"J: <b>{kdj_1d['J']:.3f}</b>\n"
        
        # MACD
        macd_1d = indicators_1d['MACD']
        report += f"MACD(12,26,9): DIF: <b>{macd_1d['DIF']:.3f}</b>  "
        report += f"DEA: <b>{macd_1d['DEA']:.3f}</b>  "
        report += f"MACD: <b>{macd_1d['MACD']:.3f}</b>\n"
        
        # Volume
        vol_1d = indicators_1d['VolumeMA']
        report += f"VOLUME: <b>{vol_1d['VOLUME']:,.3f}</b>  "
        report += f"MA5: <b>{vol_1d['MA5']:,.3f}</b>  "
        report += f"MA10: <b>{vol_1d['MA10']:,.3f}</b>\n\n"
        
        # БЛОК 2: Четырехчасовой таймфрейм
        report += "🕓 <b>ЧЕТЫРЕХЧАСОВОЙ ТАЙМФРЕЙМ (4H)</b>\n"
        
        # Stoch RSI
        stoch_4h = indicators_4h['StochRSI']
        report += f"Stoch RSI: <b>{stoch_4h['K']:.2f}</b>  <b>{stoch_4h['D']:.2f}</b>\n"
        
        # RSI multiple
        rsi_4h = indicators_4h['RSI']
        report += f"RSI6: <b>{rsi_4h['RSI6']:.2f}</b>  "
        report += f"RSI12: <b>{rsi_4h['RSI12']:.2f}</b>  "
        report += f"RSI24: <b>{rsi_4h['RSI24']:.2f}</b>\n"
        
        # Williams %R
        wr_4h = indicators_4h['WilliamsR']
        report += f"WR14: <b>{wr_4h['WR14']:.5f}</b>  "
        report += f"WR20: <b>{wr_4h['WR20']:.5f}</b>\n"
        
        # KDJ
        kdj_4h = indicators_4h['KDJ']
        report += f"KDJ(9,3,3): <b>{kdj_4h['K']:.3f}</b>  "
        report += f"D: <b>{kdj_4h['D']:.3f}</b>  "
        report += f"J: <b>{kdj_4h['J']:.3f}</b>\n"
        
        # MACD
        macd_4h = indicators_4h['MACD']
        report += f"MACD(12,26,9): DIF: <b>{macd_4h['DIF']:.3f}</b>  "
        report += f"DEA: <b>{macd_4h['DEA']:.3f}</b>  "
        report += f"MACD: <b>{macd_4h['MACD']:.3f}</b>\n"
        
        # Volume
        vol_4h = indicators_4h['VolumeMA']
        report += f"VOLUME: <b>{vol_4h['VOLUME']:,.3f}</b>  "
        report += f"MA5: <b>{vol_4h['MA5']:,.3f}</b>  "
        report += f"MA10: <b>{vol_4h['MA10']:,.3f}</b>\n\n"
        
        # БЛОК 3: Часовой таймфрейм
        report += "🕐 <b>ЧАСОВОЙ ТАЙМФРЕЙМ (1H)</b>\n"
        
        # Stoch RSI
        stoch_1h = indicators_1h['StochRSI']
        report += f"Stoch RSI: <b>{stoch_1h['K']:.2f}</b>  <b>{stoch_1h['D']:.2f}</b>\n"
        
        # RSI multiple
        rsi_1h = indicators_1h['RSI']
        report += f"RSI6: <b>{rsi_1h['RSI6']:.2f}</b>  "
        report += f"RSI12: <b>{rsi_1h['RSI12']:.2f}</b>  "
        report += f"RSI24: <b>{rsi_1h['RSI24']:.2f}</b>\n"
        
        # Williams %R
        wr_1h = indicators_1h['WilliamsR']
        report += f"WR14: <b>{wr_1h['WR14']:.5f}</b>  "
        report += f"WR20: <b>{wr_1h['WR20']:.5f}</b>\n"
        
        # KDJ
        kdj_1h = indicators_1h['KDJ']
        report += f"KDJ(9,3,3): <b>{kdj_1h['K']:.3f}</b>  "
        report += f"D: <b>{kdj_1h['D']:.3f}</b>  "
        report += f"J: <b>{kdj_1h['J']:.3f}</b>\n"
        
        # MACD
        macd_1h = indicators_1h['MACD']
        report += f"MACD(12,26,9): DIF: <b>{macd_1h['DIF']:.3f}</b>  "
        report += f"DEA: <b>{macd_1h['DEA']:.3f}</b>  "
        report += f"MACD: <b>{macd_1h['MACD']:.3f}</b>\n"
        
        # Volume
        vol_1h = indicators_1h['VolumeMA']
        report += f"VOLUME: <b>{vol_1h['VOLUME']:,.3f}</b>  "
        report += f"MA5: <b>{vol_1h['MA5']:,.3f}</b>  "
        report += f"MA10: <b>{vol_1h['MA10']:,.3f}</b>\n\n"
        
        # БЛОК 4: Пятиминутный таймфрейм
        report += "⏱ <b>ПЯТИМИНУТНЫЙ ТАЙМФРЕЙМ (5M)</b>\n"
        
        # Stoch RSI
        stoch_5m = indicators_5m['StochRSI']
        report += f"Stoch RSI: <b>{stoch_5m['K']:.2f}</b>  <b>{stoch_5m['D']:.2f}</b>\n"
        
        # RSI multiple
        rsi_5m = indicators_5m['RSI']
        report += f"RSI6: <b>{rsi_5m['RSI6']:.2f}</b>  "
        report += f"RSI12: <b>{rsi_5m['RSI12']:.2f}</b>  "
        report += f"RSI24: <b>{rsi_5m['RSI24']:.2f}</b>\n"
        
        # Williams %R
        wr_5m = indicators_5m['WilliamsR']
        report += f"WR14: <b>{wr_5m['WR14']:.5f}</b>  "
        report += f"WR20: <b>{wr_5m['WR20']:.5f}</b>\n"
        
        # KDJ
        kdj_5m = indicators_5m['KDJ']
        report += f"KDJ(9,3,3): <b>{kdj_5m['K']:.3f}</b>  "
        report += f"D: <b>{kdj_5m['D']:.3f}</b>  "
        report += f"J: <b>{kdj_5m['J']:.3f}</b>\n"
        
        # MACD
        macd_5m = indicators_5m['MACD']
        report += f"MACD(12,26,9): DIF: <b>{macd_5m['DIF']:.3f}</b>  "
        report += f"DEA: <b>{macd_5m['DEA']:.3f}</b>  "
        report += f"MACD: <b>{macd_5m['MACD']:.3f}</b>\n"
        
        # Volume
        vol_5m = indicators_5m['VolumeMA']
        report += f"VOLUME: <b>{vol_5m['VOLUME']:,.3f}</b>  "
        report += f"MA5: <b>{vol_5m['MA5']:,.3f}</b>  "
        report += f"MA10: <b>{vol_5m['MA10']:,.3f}</b>\n\n"
        
        report += f"🕒 Обновлено: {datetime.now().strftime('%H:%M:%S')}"
        
        return report

    def get_telegram_updates(self):
        """Получение обновлений из Telegram"""
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/getUpdates"
            params = {'timeout': 30, 'offset': self.last_update_id}
            
            response = requests.get(url, params=params, timeout=35)
            data = response.json()
            
            if not data['ok']:
                return []
                
            updates = data['result']
            if updates:
                self.last_update_id = updates[-1]['update_id'] + 1
                
            return updates
            
        except Exception as e:
            logger.error(f"Ошибка получения обновлений: {e}")
            return []

    def process_symbol_request(self, symbol):
        """Обработка запроса по символу"""
        try:
            # Нормализация символа (добавляем USDT если нужно)
            if not symbol.endswith('USDT'):
                symbol = symbol + 'USDT'
                
            logger.info(f"Обработка запроса для {symbol}")
            
            # Получаем индикаторы для всех четырех таймфреймов
            indicators_1d = self.get_indicators_for_timeframe(symbol, 'D')
            indicators_4h = self.get_indicators_for_timeframe(symbol, '240')
            indicators_1h = self.get_indicators_for_timeframe(symbol, '60')
            indicators_5m = self.get_indicators_for_timeframe(symbol, '5')
            
            if indicators_1d is None or indicators_4h is None or indicators_1h is None or indicators_5m is None:
                self.send_telegram_message(f"❌ Не удалось получить данные для {symbol}. Проверьте правильность названия.")
                return
                
            # Формируем и отправляем отчет
            report = self.format_indicators_report(symbol, indicators_1d, indicators_4h, indicators_1h, indicators_5m)
            self.send_telegram_message(report)
            
        except Exception as e:
            error_msg = f"❌ Ошибка обработки {symbol}: {str(e)}"
            self.send_telegram_message(error_msg)
            logger.error(error_msg)

    def run(self):
        """Запуск бота"""
        # Запускаем фоновую задачу для Colab
        self._start_colab_keep_alive()
        
        logger.info("Бот для анализа индикаторов запущен...")
        
        start_msg = "📈 <b>Бот анализа индикаторов запущен</b>\n\n"
        start_msg += "Отправьте мне название фьючерса (например: BTCUSDT или просто BTC)\n"
        start_msg += "и я пришлю полный анализ индикаторов на 4 таймфреймах!\n\n"
        start_msg += "<b>Индикаторы как в приложении биржи:</b>\n"
        start_msg += "• Stoch RSI (14,3,3)\n"
        start_msg += "• RSI (6,12,24)\n"
        start_msg += "• Williams %R (14,20)\n"
        start_msg += "• KDJ (9,3,3)\n"
        start_msg += "• MACD (12,26,9)\n"
        start_msg += "• Volume + MA5/MA10\n\n"
        start_msg += "<b>Таймфреймы:</b> 1D • 4H • 1H • 5M"
        
        self.send_telegram_message(start_msg)
        
        while True:
            try:
                updates = self.get_telegram_updates()
                
                for update in updates:
                    if 'message' in update and 'text' in update['message']:
                        message = update['message']
                        text = message['text'].strip().upper()
                        chat_id = message['chat']['id']
                        
                        # Проверяем что сообщение от нужного чата
                        if str(chat_id) == self.telegram_chat_id:
                            self.process_symbol_request(text)
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Ошибка в основном цикле: {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = FuturesIndicatorBot()
    bot.run()