import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import warnings
import time
warnings.filterwarnings('ignore')

class TradingScanner:
    def __init__(self):
        # Clean stock universe - ONLY active, liquid stocks + crypto
        self.sectors = {
            'Tech Giants': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX'],
            
            'Top Tech': ['AAPL', 'MSFT', 'NVDA', 'AMD', 'AVGO', 'QCOM', 'INTC', 'MU', 'AMAT', 'LRCX', 
                        'ASML', 'KLAC', 'CRM', 'ADBE', 'ORCL', 'NOW', 'SNOW', 'PANW', 'FTNT',
                        'INTU', 'TXN', 'ADI', 'CDNS', 'ANET', 'ZS', 'CRWD', 'DOCU', 'TEAM', 'PLTR'],
            
            'Communication': ['META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'T', 'VZ', 'TMUS',
                            'CHTR', 'CMCSA', 'WBD', 'PARA', 'MTCH', 'SNAP', 'EA', 'TTWO',
                            'ROKU', 'LYV', 'NXST', 'TME', 'BILI', 'SPOT', 'PINS', 'RBLX'],
            
            'Consumer': ['AMZN', 'TSLA', 'HD', 'LOW', 'NKE', 'MCD', 'SBUX', 'TJX', 'TGT',
                        'BKNG', 'EBAY', 'RCL', 'MAR', 'MGM', 'WYNN', 'HLT', 'CZR', 'YUM',
                        'CMG', 'F', 'GM', 'RIVN', 'NIO', 'XPEV', 'DKNG', 'ETSY', 'AZO',
                        'ORLY', 'KMX', 'ABNB', 'CCL', 'NCLH', 'LULU', 'ROST', 'DG'],
            
            'Finance': ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'AXP', 'BRK-B', 'SCHW', 'BLK',
                       'TROW', 'CME', 'ICE', 'COIN', 'BX', 'KKR', 'APO', 'V', 'MA', 'PYPL'],
            
            'Healthcare': ['LLY', 'JNJ', 'MRK', 'ABBV', 'PFE', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN',
                          'VRTX', 'UNH', 'HUM', 'CI', 'CNC', 'HCA', 'DGX', 'DHR', 'TMO', 'ISRG',
                          'MTD', 'ZBH', 'EW', 'BDX', 'MDT', 'SYK', 'BSX', 'ILMN', 'MRNA'],
            
            'Energy': ['XOM', 'CVX', 'SLB', 'OXY', 'COP', 'EOG', 'MPC', 'VLO', 'PSX', 'HES', 'HAL',
                      'DVN', 'FANG', 'APA', 'BKR', 'KMI', 'LNG', 'TRGP', 'WMB', 'ENB'],
            
            'Industrial': ['CAT', 'DE', 'BA', 'HON', 'GE', 'LMT', 'RTX', 'NOC', 'GD', 'UNP', 'NSC',
                          'CSX', 'UPS', 'FDX', 'CARR', 'OTIS', 'IR', 'MMM', 'ETN', 'EMR'],
            
            # CRYPTO SECTORS - These will definitely work
            'Crypto Major': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 
                           'DOGE-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LTC-USD', 'UNI-USD', 'LINK-USD'],
            
            'Crypto DeFi': ['UNI-USD', 'AAVE-USD', 'MKR-USD', 'COMP-USD', 'CRV-USD', 'SNX-USD', 'SUSHI-USD'],
            
            'BTC Pairs': ['BTC-USD', 'BTC-EUR', 'BTC-GBP'],
            
            'ETH Pairs': ['ETH-USD', 'ETH-EUR', 'ETH-GBP', 'ETH-BTC']
        }
        
        # Push notification configurations
        self.push_config = {
            'pushover_token': 'YOUR_PUSHOVER_APP_TOKEN',
            'pushover_user': 'YOUR_PUSHOVER_USER_KEY',
            'telegram_token': 'YOUR_TELEGRAM_BOT_TOKEN',
            'telegram_chat_id': 'YOUR_TELEGRAM_CHAT_ID',
            'pushbullet_token': 'YOUR_PUSHBULLET_ACCESS_TOKEN',
        }
    
    def get_stock_data(self, symbol, period='6mo'):
        """Fetch stock or crypto data with error handling"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if len(data) < 50:  # Need sufficient data for indicators
                return None
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average using pandas"""
        return data.ewm(span=period).mean()
    
    def calculate_stoch_rsi(self, close_prices, period=14, stoch_period=3):
        """Calculate Stochastic RSI using pandas/numpy"""
        # Calculate RSI first
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic of RSI
        rsi_low = rsi.rolling(window=period).min()
        rsi_high = rsi.rolling(window=period).max()
        stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
        
        # Smooth with moving average
        stoch_k = stoch_rsi.rolling(window=stoch_period).mean()
        stoch_d = stoch_k.rolling(window=stoch_period).mean()
        
        return stoch_k, stoch_d
    
    def calculate_macd(self, close_prices, fast=12, slow=26, signal=9):
        """Calculate MACD using pandas"""
        ema_fast = self.calculate_ema(close_prices, fast)
        ema_slow = self.calculate_ema(close_prices, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = self.calculate_ema(macd_line, signal)
        macd_hist = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_hist
    
    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range using pandas"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr

    def calculate_indicators(self, data):
        """Calculate all required technical indicators using pandas/numpy"""
        if data is None or len(data) < 50:
            return None
            
        try:
            # Price data as pandas Series
            high = data['High']
            low = data['Low']
            close = data['Close']
            volume = data['Volume']
            
            # EMAs
            ema_21 = self.calculate_ema(close, 21)
            ema_50 = self.calculate_ema(close, 50)
            ema_200 = self.calculate_ema(close, 200)
            
            # Volume EMA
            volume_ema20 = self.calculate_ema(volume, 20)
            
            # Stochastic RSI
            stoch_k, stoch_d = self.calculate_stoch_rsi(close)
            
            # MACD
            macd_line, macd_signal, macd_hist = self.calculate_macd(close)
            
            # ATR for stops
            atr = self.calculate_atr(high, low, close)
            
            indicators = {
                'ema_21': ema_21.values,
                'ema_50': ema_50.values,
                'ema_200': ema_200.values,
                'volume_ema20': volume_ema20.values,
                'stoch_k': stoch_k.values,
                'stoch_d': stoch_d.values,
                'macd_line': macd_line.values,
                'macd_signal': macd_signal.values,
                'macd_hist': macd_hist.values,
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
        """Check weekly Stoch RSI filter (20 < K < 60)"""
        try:
            daily_data = self.get_stock_data(symbol, period='2y')
            if daily_data is None:
                return False
                
            # Resample to weekly
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
    
    def analyze_stock(self, symbol, sector):
        """Analyze single stock against all strategy rules"""
        
        # Get data and indicators
        data = self.get_stock_data(symbol)
        if data is None:
            return None
            
        indicators = self.calculate_indicators(data)
        if indicators is None:
            return None
        
        # Current values (latest data point)
        current_close = indicators['close'][-1]
        current_volume = indicators['volume'][-1]
        current_high = indicators['high'][-1]
        current_low = indicators['low'][-1]
        
        # Latest indicator values
        ema_200_current = indicators['ema_200'][-1]
        ema_21_current = indicators['ema_21'][-1] 
        ema_50_current = indicators['ema_50'][-1]
        volume_ema20_current = indicators['volume_ema20'][-1]
        stoch_k_current = indicators['stoch_k'][-1]
        macd_line_current = indicators['macd_line'][-1]
        macd_signal_current = indicators['macd_signal'][-1]
        atr_current = indicators['atr'][-1]
        
        # Initialize analysis results
        analysis = {
            'symbol': symbol,
            'sector': sector,
            'current_price': current_close,
            'checks': {},
            'setup_grade': 'FAIL',
            'reason': []
        }
        
        # 1. Weekly Filter Check
        weekly_filter_pass = self.check_weekly_filter(symbol)
        analysis['checks']['weekly_stoch_rsi'] = weekly_filter_pass
        if not weekly_filter_pass:
            analysis['reason'].append('Weekly Stoch RSI not in 20-60 range')
            return analysis
        
        # 2. Trend Filter (Price > 200 EMA)
        trend_filter = current_close > ema_200_current
        analysis['checks']['trend_filter'] = trend_filter
        if not trend_filter:
            analysis['reason'].append('Price below 200 EMA')
        
        # 3. Pullback to Support (close to 21 EMA or 50 EMA)
        distance_to_21ema = abs(current_close - ema_21_current) / current_close
        distance_to_50ema = abs(current_close - ema_50_current) / current_close
        pullback_filter = distance_to_21ema <= 0.03 or distance_to_50ema <= 0.03  # Within 3%
        analysis['checks']['pullback_filter'] = pullback_filter
        if not pullback_filter:
            analysis['reason'].append('Not near 21 EMA or 50 EMA support')
        
        # 4. Momentum Reset (Stoch RSI crossed up from below 20 in last 5-8 candles)
        stoch_crossed_up = False
        for i in range(5, min(9, len(indicators['stoch_k']))):
            if (indicators['stoch_k'][-i] < 20 and 
                indicators['stoch_k'][-i+1] > indicators['stoch_k'][-i]):
                stoch_crossed_up = True
                break
        analysis['checks']['momentum_reset'] = stoch_crossed_up
        if not stoch_crossed_up:
            analysis['reason'].append('Stoch RSI did not cross up from <20 recently')
        
        # 5. MACD Confirmation (MACD line > signal line)
        macd_bullish = macd_line_current > macd_signal_current
        analysis['checks']['macd_confirmation'] = macd_bullish
        if not macd_bullish:
            analysis['reason'].append('MACD not bullish')
        
        # 6. Volume Confirmation (Volume >= 1.3x EMA20)
        volume_confirmation = current_volume >= (volume_ema20_current * 1.3)
        analysis['checks']['volume_confirmation'] = volume_confirmation
        if not volume_confirmation:
            analysis['reason'].append('Volume below 1.3x average')
        
        # 7. Candle Quality (simplified - green candle in top 30% of range)
        candle_range = current_high - current_low
        close_position = (current_close - current_low) / candle_range if candle_range > 0 else 0
        bullish_candle = close_position >= 0.7  # Close in top 30%
        analysis['checks']['candle_quality'] = bullish_candle
        if not bullish_candle:
            analysis['reason'].append('Weak candle quality')
        
        # 8. Risk-Reward Check (simplified using recent swing high and ATR stop)
        recent_high = max(indicators['high'][-20:])  # 20-day high as target
        atr_stop = current_close - (2 * atr_current)  # 2 ATR stop
        
        potential_gain = recent_high - current_close
        potential_loss = current_close - atr_stop
        rrr = potential_gain / potential_loss if potential_loss > 0 else 0
        
        rrr_acceptable = rrr >= 2.0
        analysis['checks']['risk_reward'] = rrr_acceptable
        analysis['rrr'] = round(rrr, 2)
        if not rrr_acceptable:
            analysis['reason'].append(f'RRR only {rrr:.2f}, need >= 2.0')
        
        # Determine setup grade
        all_checks = [trend_filter, pullback_filter, stoch_crossed_up, macd_bullish, 
                     volume_confirmation, bullish_candle, rrr_acceptable]
        
        if all(all_checks):
            analysis['setup_grade'] = 'A+'
            analysis['reason'] = ['All criteria met - A+ Setup!']
        elif sum(all_checks) >= 6:  # Allow one minor miss
            analysis['setup_grade'] = 'A'
            analysis['reason'].append('Minor criteria miss - A Setup')
        
        return analysis
    
    def scan_all_sectors(self):
        """Scan all stocks across all sectors"""
        results = {
            'A+': [],
            'A': [],
            'scan_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_scanned': 0,
            'sector_breakdown': {}
        }
        
        for sector, symbols in self.sectors.items():
            print(f"Scanning {sector} sector...")
            sector_results = {'A+': 0, 'A': 0, 'total': 0}
            
            for symbol in symbols:
                try:
                    analysis = self.analyze_stock(symbol, sector)
                    if analysis:
                        results['total_scanned'] += 1
                        sector_results['total'] += 1
                        
                        if analysis['setup_grade'] == 'A+':
                            results['A+'].append(analysis)
                            sector_results['A+'] += 1
                        elif analysis['setup_grade'] == 'A':
                            results['A'].append(analysis)
                            sector_results['A'] += 1
                
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")
                    continue
            
            results['sector_breakdown'][sector] = sector_results
            print(f"{sector}: {sector_results['A+']} A+ setups, {sector_results['A']} A setups")
        
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
    
    def send_pushbullet_notification(self, title, message):
        """Send push notification via Pushbullet"""
        try:
            if self.push_config['pushbullet_token'] != 'YOUR_PUSHBULLET_ACCESS_TOKEN':
                
                url = 'https://api.pushbullet.com/v2/pushes'
                headers = {
                    'Access-Token': self.push_config['pushbullet_token'],
                    'Content-Type': 'application/json'
                }
                data = {
                    'type': 'note',
                    'title': title,
                    'body': message
                }
                
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    print("✅ Pushbullet notification sent!")
                    return True
                else:
                    print(f"❌ Pushbullet error: {response.status_code}")
        except Exception as e:
            print(f"❌ Pushbullet error: {e}")
        return False
    
    def format_mobile_alert(self, results):
        """Format compact message for mobile notifications"""
        a_plus_count = len(results['A+'])
        a_count = len(results['A'])
        
        if a_plus_count == 0 and a_count == 0:
            return "📊 No qualifying setups found today", "No setups found"
        
        title = f"🚨 {a_plus_count} A+ Setups Found!"
        
        message = f"Trading Alert - {results['scan_time']}\n\n"
        
        if a_plus_count > 0:
            message += f"🎯 A+ SETUPS ({a_plus_count}):\n"
            for setup in results['A+'][:5]:
                message += f"• {setup['symbol']} ${setup['current_price']:.2f} (RRR:{setup.get('rrr', 'N/A')})\n"
            
            if a_plus_count > 5:
                message += f"• ...and {a_plus_count - 5} more\n"
        
        if a_count > 0:
            message += f"\n⭐ A SETUPS ({a_count}):\n"
            for setup in results['A'][:3]:
                message += f"• {setup['symbol']} ${setup['current_price']:.2f}\n"
            
            if a_count > 3:
                message += f"• ...and {a_count - 3} more\n"
        
        return title, message
    
    def run_daily_scan(self):
        """Main function to run daily scan and send alerts"""
        print("🔍 Starting Daily Trading Scan...")
        print("=" * 50)
        
        # Run the scan
        results = self.scan_all_sectors()
        
        # Print results to console
        print(f"\n📊 SCAN COMPLETE - {results['scan_time']}")
        print(f"Total Assets Scanned: {results['total_scanned']}")
        print(f"A+ Setups Found: {len(results['A+'])}")
        print(f"A Setups Found: {len(results['A'])}")
        
        # Send mobile alerts if setups found
        if len(results['A+']) > 0 or len(results['A']) > 0:
            print("\n📱 Sending mobile alerts...")
            title, mobile_message = self.format_mobile_alert(results)
            
            # Try multiple notification services
            success = False
            
            if not success:
                success = self.send_pushover_notification(title, mobile_message)
            
            if not success:
                success = self.send_telegram_notification(f"*{title}*\n\n{mobile_message}")
            
            if not success:
                success = self.send_pushbullet_notification(title, mobile_message)
            
            if not success:
                print("❌ No notification services configured. Please set up at least one.")
        else:
            print("\n😴 No qualifying setups found today")
        
        return results

# Usage Example
if __name__ == "__main__":
    # Initialize scanner
    scanner = TradingScanner()
    
    # OPTIONAL: Set up notifications (choose one)
    # scanner.push_config['pushover_token'] = 'YOUR_PUSHOVER_APP_TOKEN'
    # scanner.push_config['pushover_user'] = 'u9uzeesuxqax45yhjrzyfvv6opbm6y'
    
    # scanner.push_config['telegram_token'] = 'YOUR_BOT_TOKEN'
    # scanner.push_config['telegram_chat_id'] = 'YOUR_CHAT_ID'
    
    # Run the daily scan
    results = scanner.run_daily_scan()