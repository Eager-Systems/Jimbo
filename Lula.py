import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import warnings
import time
warnings.filterwarnings('ignore')

class LulaCryptoScanner:
    def __init__(self):
        # Crypto-only universe - optimized for crypto trading
        self.sectors = {
            'Crypto Major': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 
                           'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LTC-USD', 'UNI-USD', 'LINK-USD'],
            
            'Crypto DeFi': ['UNI-USD', 'AAVE-USD', 'MKR-USD', 'CRV-USD', 'SNX-USD', 'SUSHI-USD'],
            
            'BTC Pairs': ['BTC-USD', 'BTC-EUR', 'BTC-GBP', 'ETH-BTC', 'ADA-BTC', 'ATOM-BTC', 'AVAX-BTC', 
                         'BAT-BTC', 'BCH-BTC', 'COMP-BTC', 'CRV-BTC', 'DASH-BTC', 'DOGE-BTC', 'DOT-BTC',
                         'FIL-BTC', 'GRT-BTC', 'LINK-BTC', 'LRC-BTC', 'MATIC-BTC', 'MKR-BTC',
                         'SNX-BTC', 'SOL-BTC', 'UNI-BTC', 'XLM-BTC', 'XTZ-BTC', 'YFI-BTC'],
            
            'ETH Pairs': ['ETH-USD', 'ETH-EUR', 'ETH-GBP', 'ETH-BTC', 'AAVE-ETH', 'ADA-ETH', 'ALGO-ETH',
                         'ATOM-ETH', 'BCH-ETH', 'DOT-ETH', 'ETC-ETH', 'FIL-ETH', 'LTC-ETH', 'SOL-ETH',
                         'TRX-ETH', 'UNI-ETH', 'XRP-ETH']
        }
        
        # Position sizing configuration
        self.position_config = {
            'base_size': 0.25,      # Base position size (0.25%)
            'upgrade_size': 0.5,    # Upgraded position size (0.5%)
        }
        
        # Exchange API configurations
        self.exchange_config = {
            'coinbase_url': 'https://api.exchange.coinbase.com',
            'kraken_url': 'https://api.kraken.com/0/public',
            'rate_limit': 1.0
        }
        
        # Push notification configurations
        self.push_config = {
            'pushover_token': 'YOUR_PUSHOVER_APP_TOKEN',
            'pushover_user': 'YOUR_PUSHOVER_USER_KEY',
            'telegram_token': 'YOUR_TELEGRAM_BOT_TOKEN',
            'telegram_chat_id': 'YOUR_TELEGRAM_CHAT_ID',
        }
    
    def get_coinbase_data(self, pair, period=180):
        """Fetch crypto data from Coinbase Pro API"""
        try:
            url = f"{self.exchange_config['coinbase_url']}/products/{pair}/candles"
            
            params = {
                'granularity': 86400,
                'start': (datetime.now() - timedelta(days=period)).isoformat(),
                'end': datetime.now().isoformat()
            }
            
            response = requests.get(url, params=params)
            time.sleep(self.exchange_config['rate_limit'])
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=['timestamp', 'Low', 'High', 'Open', 'Close', 'Volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                
                return df if len(df) >= 50 else None
            else:
                return None
                
        except Exception as e:
            print(f"Error fetching Coinbase data for {pair}: {e}")
            return None
    
    def get_kraken_data(self, pair, period=180):
        """Fetch crypto data from Kraken API"""
        try:
            kraken_pair = pair.replace('-', '')
            if 'ETH' in kraken_pair and kraken_pair != 'ETH':
                if kraken_pair.endswith('ETH'):
                    kraken_pair = kraken_pair
                elif kraken_pair.startswith('ETH'):
                    if 'USD' in kraken_pair:
                        kraken_pair = 'ETHUSD'
                    elif 'EUR' in kraken_pair:
                        kraken_pair = 'ETHEUR'
                    elif 'BTC' in kraken_pair:
                        kraken_pair = 'ETHXBT'
            
            url = f"{self.exchange_config['kraken_url']}/OHLC"
            params = {'pair': kraken_pair, 'interval': 1440}
            
            response = requests.get(url, params=params)
            time.sleep(self.exchange_config['rate_limit'])
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('error') or not data.get('result'):
                    return None
                    
                pair_data = list(data['result'].values())[0]
                df = pd.DataFrame(pair_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'vwap', 'Volume', 'count'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
                
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = df[col].astype(float)
                
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                cutoff_date = datetime.now() - timedelta(days=period)
                df = df[df.index >= cutoff_date]
                
                return df if len(df) >= 50 else None
            else:
                return None
                
        except Exception as e:
            print(f"Error fetching Kraken data for {pair}: {e}")
            return None

    def get_crypto_data(self, symbol, period='6mo'):
        """Smart routing for crypto data across exchanges"""
        try:
            sector = None
            for sector_name, symbols in self.sectors.items():
                if symbol in symbols:
                    sector = sector_name
                    break
            
            if sector == 'BTC Pairs' and '-BTC' in symbol:
                return self.get_coinbase_data(symbol)
            elif sector == 'ETH Pairs' and '-ETH' in symbol:
                return self.get_kraken_data(symbol)
            else:
                stock = yf.Ticker(symbol)
                data = stock.history(period=period)
                return data if len(data) >= 50 else None
                
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    def calculate_stoch_rsi(self, close_prices, period=14, stoch_period=3):
        """Calculate Stochastic RSI"""
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi_low = rsi.rolling(window=period).min()
        rsi_high = rsi.rolling(window=period).max()
        stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
        
        stoch_k = stoch_rsi.rolling(window=stoch_period).mean()
        stoch_d = stoch_k.rolling(window=stoch_period).mean()
        
        return stoch_k, stoch_d
    
    def calculate_macd(self, close_prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = self.calculate_ema(close_prices, fast)
        ema_slow = self.calculate_ema(close_prices, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd_line, signal)
        macd_hist = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_hist
    
    def calculate_macd_buy_sell(self, macd_line, macd_signal):
        """Calculate MACD Buy/Sell signals"""
        try:
            buy_sell_signals = []
            
            for i in range(len(macd_line)):
                if i == 0:
                    buy_sell_signals.append(0)
                    continue
                    
                prev_macd = macd_line[i-1] if not pd.isna(macd_line[i-1]) else 0
                current_macd = macd_line[i] if not pd.isna(macd_line[i]) else 0
                prev_signal = macd_signal[i-1] if not pd.isna(macd_signal[i-1]) else 0
                current_signal = macd_signal[i] if not pd.isna(macd_signal[i]) else 0
                
                if prev_macd <= prev_signal and current_macd > current_signal:
                    buy_sell_signals.append(1)  # BUY signal
                elif prev_macd >= prev_signal and current_macd < current_signal:
                    buy_sell_signals.append(-1)  # SELL signal
                else:
                    buy_sell_signals.append(0)  # No signal
            
            return buy_sell_signals
            
        except Exception as e:
            print(f"Error calculating MACD Buy/Sell: {e}")
            return [0] * len(macd_line)
    
    def calculate_steves_dc_macd(self, close, high, low, volume, ema_50, ema_200):
        """Calculate Steve's DC-MACD with crypto-appropriate filters"""
        try:
            macd_line, macd_signal, macd_hist = self.calculate_macd(pd.Series(close))
            
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            dc_macd_signals = []
            
            for i in range(len(close)):
                if i < 50:
                    dc_macd_signals.append(0)
                    continue
                    
                current_price = close[i]
                prev_macd = macd_line.iloc[i-1] if i > 0 and not pd.isna(macd_line.iloc[i-1]) else 0
                current_macd = macd_line.iloc[i] if not pd.isna(macd_line.iloc[i]) else 0
                prev_signal = macd_signal.iloc[i-1] if i > 0 and not pd.isna(macd_signal.iloc[i-1]) else 0
                current_signal = macd_signal.iloc[i] if not pd.isna(macd_signal.iloc[i]) else 0
                current_hist = macd_hist.iloc[i] if not pd.isna(macd_hist.iloc[i]) else 0
                prev_hist = macd_hist.iloc[i-1] if i > 0 and not pd.isna(macd_hist.iloc[i-1]) else 0
                current_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
                
                # Steve's DC-MACD BUY Signal (crypto-optimized)
                macd_bullish_cross = (prev_macd <= prev_signal and current_macd > current_signal)
                histogram_positive_or_growing = (current_hist > 0 or current_hist > prev_hist)
                price_above_trend = current_price > ema_200[i] if not pd.isna(ema_200[i]) else False
                momentum_suitable = 20 < current_rsi < 85  # Wider range for crypto
                
                if (macd_bullish_cross and histogram_positive_or_growing and 
                    price_above_trend and momentum_suitable):
                    dc_macd_signals.append(1)  # BUY signal
                else:
                    dc_macd_signals.append(0)  # No signal
            
            return dc_macd_signals
            
        except Exception as e:
            print(f"Error calculating Steve's DC-MACD: {e}")
            return [0] * len(close)
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range for crypto (wider multiplier)"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr

    def calculate_indicators(self, data):
        """Calculate all technical indicators for crypto"""
        if data is None or len(data) < 50:
            return None
            
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            volume = data['Volume']
            
            # EMAs
            ema_21 = self.calculate_ema(close, 21)
            ema_50 = self.calculate_ema(close, 50)
            ema_200 = self.calculate_ema(close, 200)
            
            # Stochastic RSI
            stoch_k, stoch_d = self.calculate_stoch_rsi(close)
            
            # MACD components
            macd_line, macd_signal, macd_hist = self.calculate_macd(close)
            
            # MACD Buy/Sell signals
            macd_buy_sell = self.calculate_macd_buy_sell(macd_line.values, macd_signal.values)
            
            # Steve's DC-MACD (crypto-optimized)
            dc_macd_signals = self.calculate_steves_dc_macd(close.values, high.values, low.values, 
                                                          volume.values, ema_50.values, ema_200.values)
            
            # ATR with crypto-appropriate multiplier
            atr = self.calculate_atr(high, low, close)
            
            indicators = {
                'ema_21': ema_21.values,
                'ema_50': ema_50.values,
                'ema_200': ema_200.values,
                'stoch_k': stoch_k.values,
                'stoch_d': stoch_d.values,
                'macd_line': macd_line.values,
                'macd_signal': macd_signal.values,
                'macd_hist': macd_hist.values,
                'macd_buy_sell': macd_buy_sell,
                'dc_macd_signals': dc_macd_signals,
                'atr': atr.values,
                'high': high.values,
                'low': low.values,
                'close': close.values,
                'volume': volume.values
            }
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return None
    
    def check_weekly_filter(self, symbol):
        """Check weekly Stoch RSI filter"""
        try:
            daily_data = self.get_crypto_data(symbol, period='2y')
            if daily_data is None:
                return False
                
            weekly_data = daily_data.resample('W').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            
            if len(weekly_data) < 20:
                return False
                
            weekly_close = weekly_data['Close']
            weekly_stoch_k, _ = self.calculate_stoch_rsi(weekly_close)
            
            latest_weekly_stoch = weekly_stoch_k.iloc[-1]
            return 20 < latest_weekly_stoch < 60
            
        except Exception as e:
            print(f"Error checking weekly filter for {symbol}: {e}")
            return False
    
    def analyze_crypto(self, symbol, sector):
        """Analyze crypto against Lula's crypto-optimized rules with DC-MACD tier upgrade"""
        
        data = self.get_crypto_data(symbol)
        if data is None:
            return None
            
        indicators = self.calculate_indicators(data)
        if indicators is None:
            return None
        
        # Current values
        current_close = indicators['close'][-1]
        current_high = indicators['high'][-1]
        current_low = indicators['low'][-1]
        
        # Latest indicator values
        ema_200_current = indicators['ema_200'][-1]
        stoch_k_current = indicators['stoch_k'][-1]
        macd_line_current = indicators['macd_line'][-1]
        macd_signal_current = indicators['macd_signal'][-1]
        macd_buy_sell_current = indicators['macd_buy_sell'][-1]
        dc_macd_current = indicators['dc_macd_signals'][-1]
        atr_current = indicators['atr'][-1]
        
        # Initialize analysis
        analysis = {
            'symbol': symbol,
            'sector': sector,
            'current_price': current_close,
            'checks': {},
            'setup_grade': 'FAIL',
            'reason': [],
            'position_size': 0,
            'upgrade_applied': False
        }
        
        # LULA CRYPTO STRATEGY RULES
        
        # 1. Weekly Filter Check (ADVISORY - provides context, not mandatory)
        weekly_filter_pass = self.check_weekly_filter(symbol)
        analysis['checks']['weekly_stoch_rsi'] = weekly_filter_pass
        weekly_context = "favorable" if weekly_filter_pass else "extended"
        
        # Continue analysis regardless of weekly filter result
        
        # 2. Trend Filter (Price > 200 EMA) (MANDATORY)
        trend_filter = current_close > ema_200_current
        analysis['checks']['trend_filter'] = trend_filter
        if not trend_filter:
            analysis['reason'].append('Price below 200 EMA')
        
        # 3. Momentum Reset (Stoch RSI crossed up from below 20) (MANDATORY)
        stoch_crossed_up = False
        stoch_k_values = indicators['stoch_k']
        valid_stoch = [x for x in stoch_k_values if not pd.isna(x)]
        
        if len(valid_stoch) >= 8:
            for i in range(5, min(9, len(valid_stoch))):
                previous_value = valid_stoch[-(i+1)]
                current_value = valid_stoch[-i]
                
                if (previous_value < 20 and current_value > previous_value and current_value > 20):
                    stoch_crossed_up = True
                    break
        
        analysis['checks']['momentum_reset'] = stoch_crossed_up
        if not stoch_crossed_up:
            analysis['reason'].append('Stoch RSI did not cross up from <20 recently')
        
        # 4. MACD Confirmation (crypto-optimized approach)
        macd_bullish = macd_line_current > macd_signal_current
        macd_buy_signal = macd_buy_sell_current == 1
        steves_dc_macd_bullish = dc_macd_current == 1
        
        # Primary requirement: Either current MACD bullish OR recent MACD buy signal
        macd_confirmation = macd_bullish or macd_buy_signal
        
        analysis['checks']['macd_confirmation'] = macd_confirmation
        analysis['checks']['macd_buy_signal'] = macd_buy_signal
        analysis['checks']['steves_dc_macd'] = steves_dc_macd_bullish
        
        if not macd_confirmation:
            analysis['reason'].append('MACD not bullish and no recent buy signal')
        
        # 5. Lower RRR Requirement: 1.5:1 instead of 2:1 (crypto moves faster)
        recent_high = max(indicators['high'][-20:])
        atr_stop = current_close - (3 * atr_current)  # Wider stops for crypto volatility
        
        potential_gain = recent_high - current_close
        potential_loss = current_close - atr_stop
        rrr = potential_gain / potential_loss if potential_loss > 0 else 0
        
        rrr_acceptable = rrr >= 1.5  # Lower requirement for crypto
        analysis['checks']['risk_reward'] = rrr_acceptable
        analysis['rrr'] = round(rrr, 2)
        if not rrr_acceptable:
            analysis['reason'].append(f'RRR only {rrr:.2f}, need >= 1.5')
        
        # POSITION SIZING WITH STEVE'S DC-MACD TIER UPGRADE
        all_checks = [trend_filter, stoch_crossed_up, macd_confirmation, rrr_acceptable]
        
        # CRITICAL: Momentum reset is mandatory
        if not stoch_crossed_up:
            analysis['setup_grade'] = 'FAIL'
            analysis['reason'].append('FAILED: Momentum reset is mandatory')
            analysis['position_size'] = 0
        elif all(all_checks):
            # Base trigger: MACD bullish crossover confirmed
            if macd_buy_signal or macd_bullish:
                analysis['position_size'] = self.position_config['base_size']  # 0.25% base size
                upgrade_reason = ""
                
                # Tier upgrade: Size up if Steve's DC-MACD is also green
                if steves_dc_macd_bullish:
                    analysis['position_size'] = self.position_config['upgrade_size']  # 0.5% upgraded size
                    analysis['upgrade_applied'] = True
                    upgrade_reason = f" + DC-MACD UPGRADE ({self.position_config['upgrade_size']}% size)"
                else:
                    upgrade_reason = f" (base {self.position_config['base_size']}% size)"
                
                # Update setup grade with sizing info
                if steves_dc_macd_bullish and weekly_filter_pass:
                    analysis['setup_grade'] = 'A+'
                    analysis['reason'] = [f'All criteria + DC-MACD tier upgrade + {weekly_context} weekly timing{upgrade_reason}']
                elif steves_dc_macd_bullish:
                    analysis['setup_grade'] = 'A+'
                    analysis['reason'] = [f'All criteria + DC-MACD tier upgrade (weekly: {weekly_context}){upgrade_reason}']
                elif weekly_filter_pass:
                    analysis['setup_grade'] = 'A+'
                    analysis['reason'] = [f'All criteria + {weekly_context} weekly timing{upgrade_reason}']
                else:
                    analysis['setup_grade'] = 'A+'
                    analysis['reason'] = [f'All criteria met (weekly: {weekly_context}){upgrade_reason}']
            else:
                analysis['setup_grade'] = 'FAIL'
                analysis['reason'].append('MACD base trigger not confirmed')
                analysis['position_size'] = 0
        elif sum(all_checks) >= 3:
            analysis['setup_grade'] = 'A'
            analysis['position_size'] = self.position_config['base_size'] * 0.5  # Reduced size for A setup
            analysis['reason'].append(f'Minor criteria miss (weekly: {weekly_context}) - Lula Crypto A Setup (reduced size)')
        else:
            analysis['setup_grade'] = 'FAIL'
            analysis['position_size'] = 0
        
        return analysis
    
    def scan_all_crypto(self):
        """Scan all crypto sectors"""
        results = {
            'A+': [],
            'A': [],
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_scanned': 0,
            'sector_breakdown': {},
            'total_upgraded_positions': 0,
            'total_base_positions': 0
        }
        
        for sector, symbols in self.sectors.items():
            print(f"Scanning {sector} sector...")
            sector_results = {'A+': 0, 'A': 0, 'total': 0, 'upgrades': 0}
            
            for symbol in symbols:
                try:
                    analysis = self.analyze_crypto(symbol, sector)
                    if analysis:
                        results['total_scanned'] += 1
                        sector_results['total'] += 1
                        
                        if analysis['setup_grade'] == 'A+':
                            results['A+'].append(analysis)
                            sector_results['A+'] += 1
                            if analysis['upgrade_applied']:
                                results['total_upgraded_positions'] += 1
                                sector_results['upgrades'] += 1
                            else:
                                results['total_base_positions'] += 1
                        elif analysis['setup_grade'] == 'A':
                            results['A'].append(analysis)
                            sector_results['A'] += 1
                            results['total_base_positions'] += 1
                
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
                    continue
                
                # Rate limiting for APIs
                if '-BTC' in symbol or '-ETH' in symbol:
                    time.sleep(0.1)
            
            results['sector_breakdown'][sector] = sector_results
            print(f"{sector}: {sector_results['A+']} A+ setups ({sector_results['upgrades']} upgrades), {sector_results['A']} A setups")
        
        return results
    
    def send_pushover_notification(self, title, message):
        """Send push notification via Pushover"""
        try:
            if (self.push_config['pushover_token'] != 'YOUR_PUSHOVER_APP_TOKEN' and 
                self.push_config['pushover_user'] != 'YOUR_PUSHOVER_USER_KEY'):
                
                url = 'https://api.pushover.net/1/messages.json'
                data = {
                    'token': self.push_config['pushover_token'],
                    'user': self.push_config['pushover_user'],
                    'title': title,
                    'message': message,
                    'priority': 1,
                    'sound': 'cashregister'
                }
                
                response = requests.post(url, data=data)
                if response.status_code == 200:
                    print("✅ Pushover notification sent!")
                    return True
                else:
                    print(f"❌ Pushover error: {response.status_code}")
        except Exception as e:
            print(f"❌ Pushover error: {e}")
        return False
    
    def send_telegram_notification(self, message):
        """Send push notification via Telegram Bot"""
        try:
            if (self.push_config['telegram_token'] != 'YOUR_TELEGRAM_BOT_TOKEN' and 
                self.push_config['telegram_chat_id'] != 'YOUR_TELEGRAM_CHAT_ID'):
                
                url = f"https://api.telegram.org/bot{self.push_config['telegram_token']}/sendMessage"
                data = {
                    'chat_id': self.push_config['telegram_chat_id'],
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                
                response = requests.post(url, data=data)
                if response.status_code == 200:
                    print("✅ Telegram notification sent!")
                    return True
                else:
                    print(f"❌ Telegram error: {response.status_code}")
        except Exception as e:
            print(f"❌ Telegram error: {e}")
        return False
    
    def format_crypto_alert(self, results):
        """Format compact message for crypto notifications with tier upgrade info"""
        a_plus_count = len(results['A+'])
        a_count = len(results['A'])
        upgraded_count = results['total_upgraded_positions']
        
        if a_plus_count == 0 and a_count == 0:
            return "📊 No crypto setups found today", "No crypto setups found"
        
        title = f"🚀 Lula Crypto: {a_plus_count} A+ Setups ({upgraded_count} Upgrades)!"
        
        message = f"Lula Crypto Alert - {results['scan_time']}\n\n"
        
        if a_plus_count > 0:
            message += f"🎯 CRYPTO A+ SETUPS ({a_plus_count}):\n"
            for setup in results['A+'][:5]:
                size_indicator = "🔥 0.5%" if setup['upgrade_applied'] else "📈 0.25%"
                upgrade_note = " (DC-MACD)" if setup['upgrade_applied'] else ""
                message += f"• {setup['symbol']} ${setup['current_price']:.4f} {size_indicator}{upgrade_note} (RRR:{setup.get('rrr', 'N/A')})\n"
            
            if a_plus_count > 5:
                message += f"• ...and {a_plus_count - 5} more\n"
        
        if a_count > 0:
            message += f"\n⭐ CRYPTO A SETUPS ({a_count}):\n"
            for setup in results['A'][:3]:
                message += f"• {setup['symbol']} ${setup['current_price']:.4f} 📈 {setup['position_size']:.2f}%\n"
            
            if a_count > 3:
                message += f"• ...and {a_count - 3} more\n"
        
        # Position sizing summary
        message += f"\n💰 POSITION SIZING:\n"
        message += f"• Base positions (0.25%): {results['total_base_positions']}\n"
        message += f"• DC-MACD upgrades (0.5%): {upgraded_count}\n"
        
        # Risk reminder for crypto
        message += f"\n⚠️ Crypto Risk: Stick to position sizing rules!"
        message += f"\n🎯 Base trigger: MACD bullish crossover"
        message += f"\n🔥 Size upgrade: + Steve's DC-MACD green"
        
        return title, message
    
    def run_crypto_scan(self):
        """Main function to run Lula crypto scan and send alerts"""
        print("🚀 Starting Lula Crypto Trading Scan with DC-MACD Tier Upgrade...")
        print("=" * 60)
        
        results = self.scan_all_crypto()
        
        print(f"\n📊 LULA CRYPTO SCAN COMPLETE - {results['scan_time']}")
        print(f"Total Crypto Assets Scanned: {results['total_scanned']}")
        print(f"A+ Setups Found: {len(results['A+'])}")
        print(f"A Setups Found: {len(results['A'])}")
        print(f"🔥 DC-MACD Tier Upgrades: {results['total_upgraded_positions']}")
        print(f"📈 Base Position Setups: {results['total_base_positions']}")
        print(f"\n💡 Position Sizing Rules:")
        print(f"   • Base size (0.25%): MACD bullish crossover confirmed")
        print(f"   • Upgrade size (0.5%): Base trigger + Steve's DC-MACD green")
        
        # Show detailed breakdown if setups found
        if len(results['A+']) > 0:
            print(f"\n🎯 A+ SETUPS BREAKDOWN:")
            for setup in results['A+']:
                size_note = f"0.5% (DC-MACD upgrade)" if setup['upgrade_applied'] else f"0.25% (base size)"
                print(f"   • {setup['symbol']}: ${setup['current_price']:.4f} - {size_note}")
        
        if len(results['A']) > 0:
            print(f"\n⭐ A SETUPS BREAKDOWN:")
            for setup in results['A']:
                print(f"   • {setup['symbol']}: ${setup['current_price']:.4f} - {setup['position_size']:.2f}% (reduced A setup)")
        
        if len(results['A+']) > 0 or len(results['A']) > 0:
            print("\n📱 Sending Lula crypto alerts...")
            title, mobile_message = self.format_crypto_alert(results)
            
            success = False
            
            if not success:
                success = self.send_pushover_notification(title, mobile_message)
            
            if not success:
                success = self.send_telegram_notification(f"*{title}*\n\n{mobile_message}")
            
            if not success:
                print("❌ No notification services configured. Please set up at least one.")
        else:
            print("\n😴 Lula found no qualifying crypto setups today")
        
        return results

    def analyze_single_crypto(self, symbol):
        """Analyze a single crypto for detailed breakdown"""
        print(f"\n🔍 DETAILED ANALYSIS: {symbol}")
        print("=" * 50)
        
        # Find sector
        sector = None
        for sector_name, symbols in self.sectors.items():
            if symbol in symbols:
                sector = sector_name
                break
        
        if not sector:
            print(f"❌ {symbol} not found in crypto universe")
            return None
        
        analysis = self.analyze_crypto(symbol, sector)
        
        if not analysis:
            print(f"❌ Could not analyze {symbol}")
            return None
        
        print(f"Symbol: {analysis['symbol']}")
        print(f"Sector: {analysis['sector']}")
        print(f"Current Price: ${analysis['current_price']:.4f}")
        print(f"Setup Grade: {analysis['setup_grade']}")
        print(f"Position Size: {analysis['position_size']:.2f}%")
        print(f"DC-MACD Upgrade: {'✅ YES' if analysis['upgrade_applied'] else '❌ NO'}")
        
        print(f"\n📋 CRITERIA CHECKLIST:")
        print(f"✅ Trend Filter (>200 EMA): {'PASS' if analysis['checks']['trend_filter'] else 'FAIL'}")
        print(f"✅ Momentum Reset (Stoch RSI): {'PASS' if analysis['checks']['momentum_reset'] else 'FAIL'}")
        print(f"✅ MACD Confirmation: {'PASS' if analysis['checks']['macd_confirmation'] else 'FAIL'}")
        print(f"✅ MACD Buy Signal: {'YES' if analysis['checks']['macd_buy_signal'] else 'NO'}")
        print(f"✅ Steve's DC-MACD: {'GREEN' if analysis['checks']['steves_dc_macd'] else 'NOT GREEN'}")
        print(f"✅ Risk/Reward (≥1.5): {'PASS' if analysis['checks']['risk_reward'] else 'FAIL'} ({analysis.get('rrr', 'N/A')})")
        print(f"✅ Weekly Stoch RSI: {'FAVORABLE' if analysis['checks']['weekly_stoch_rsi'] else 'EXTENDED'}")
        
        print(f"\n💭 REASONING:")
        for reason in analysis['reason']:
            print(f"   • {reason}")
        
        print(f"\n💰 POSITION SIZING LOGIC:")
        if analysis['setup_grade'] in ['A+', 'A']:
            if analysis['checks']['macd_buy_signal'] or analysis['checks']['macd_confirmation']:
                print(f"   • Base trigger: MACD bullish crossover ✅")
                if analysis['checks']['steves_dc_macd']:
                    print(f"   • Tier upgrade: Steve's DC-MACD green ✅")
                    print(f"   • Final size: {self.position_config['upgrade_size']}% (UPGRADED)")
                else:
                    print(f"   • Tier upgrade: Steve's DC-MACD not green ❌")
                    print(f"   • Final size: {self.position_config['base_size']}% (BASE)")
            else:
                print(f"   • Base trigger: MACD not confirmed ❌")
                print(f"   • Final size: 0% (NO ENTRY)")
        else:
            print(f"   • Setup failed mandatory criteria")
            print(f"   • Final size: 0% (NO ENTRY)")
        
        return analysis

# Usage Examples
if __name__ == "__main__":
    # Initialize Lula Crypto Scanner
    scanner = LulaCryptoScanner()
    
    # OPTIONAL: Set up notifications
    scanner.push_config['pushover_token'] = 'afh8vfyctjm48ta44od7a2w19e52t6'
    scanner.push_config['pushover_user'] = 'u9uzeesuxqax45yhjrzyfvv6opbm6y'
    
    # scanner.push_config['telegram_token'] = 'your_bot_token'
    # scanner.push_config['telegram_chat_id'] = 'your_chat_id'
    
    # OPTIONAL: Customize position sizing
    # scanner.position_config['base_size'] = 0.3      # 0.3% base size
    # scanner.position_config['upgrade_size'] = 0.6   # 0.6% upgrade size
    
    print("🎯 LULA CRYPTO STRATEGY v2.0 - DC-MACD TIER UPGRADE")
    print("=" * 60)
    print("📋 STRATEGY RULES:")
    print("   1. Trend Filter: Price > 200 EMA (MANDATORY)")
    print("   2. Momentum Reset: Stoch RSI crossed up from <20 (MANDATORY)")
    print("   3. MACD Confirmation: Bullish crossover (MANDATORY)")
    print("   4. Risk/Reward: ≥1.5:1 (MANDATORY)")
    print("   5. Weekly Stoch RSI: Advisory context only")
    print("")
    print("💰 POSITION SIZING:")
    print(f"   • Base size ({scanner.position_config['base_size']}%): MACD bullish crossover")
    print(f"   • Upgrade size ({scanner.position_config['upgrade_size']}%): Base + Steve's DC-MACD green")
    print("")
    
    # Example 1: Run full scan
    print("🚀 Running full crypto scan...")
    results = scanner.run_crypto_scan()
    
    # Example 2: Analyze specific crypto
    # print("\n" + "="*60)
    # scanner.analyze_single_crypto('BTC-USD')
    
    # Example 3: Analyze multiple specific cryptos
    # test_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']
    # for symbol in test_symbols:
    #     scanner.analyze_single_crypto(symbol)
    #     print("\n" + "-"*50)