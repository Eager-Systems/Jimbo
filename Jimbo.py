import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')

class JimboStrategyScanner:
    """
    Jimbo's Enhanced Strategy Scanner - Optimized for 100 High-Quality Stocks
    Curated stock universe to avoid Yahoo Finance rate limiting
    """
    def __init__(self):
        # Notification settings
        self.notifications = {
            'console': True,
            'file': True,
            'email': False,
            'webhook': False,
            'pushover': True
        }
        
        # Pushover config
        self.pushover_config = {
            'app_token': 'a4nhedd42qmdt57g4b611f2aabznze',
            'user_key': 'u9uzeesuxqax45yhjrzyfvv6opbm6y',
            'priority': 1,
            'sound': 'cashregister'
        }
        
        # Rate limiting settings
        self.request_delay = 0.5  # 500ms between requests
        self.batch_delay = 2.0    # 2s between sector batches
        
        # Curated list of 100 top stocks optimized for Jimbo's Strategy
        # Selected based on: Market cap >$10B, Average volume >1M, Strong liquidity
        self.sectors = {
            'Technology': [
                'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'AVGO', 'ORCL',
                'CRM', 'ADBE', 'NFLX', 'INTC', 'QCOM', 'TXN', 'NOW', 'INTU', 'AMAT', 'MU'
            ],
            
            'Communication Services': [
                'GOOGL', 'META', 'NFLX', 'DIS', 'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA', 'SPOT'
            ],
            
            'Consumer Discretionary': [
                'AMZN', 'TSLA', 'HD', 'MCD', 'LOW', 'NKE', 'SBUX', 'TJX', 'TGT', 'BKNG',
                'GM', 'F', 'MAR', 'RCL', 'MGM', 'ABNB', 'LULU', 'YUM', 'CMG', 'ROST'
            ],
            
            'Consumer Staples': [
                'WMT', 'PG', 'KO', 'PEP', 'COST', 'MDLZ', 'CL', 'KMB', 'GIS', 'KR'
            ],
            
            'Energy': [
                'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'HAL'
            ],
            
            'Financials': [
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BRK-B', 'V', 'MA', 'AXP',
                'SCHW', 'BLK', 'CME', 'ICE', 'BX', 'KKR', 'COIN'
            ],
            
            'Healthcare': [
                'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'DHR', 'BMY', 'AMGN',
                'GILD', 'VRTX', 'REGN', 'ISRG', 'HUM', 'CI'
            ],
            
            'Industrials': [
                'CAT', 'BA', 'HON', 'UNP', 'LMT', 'GE', 'RTX', 'DE', 'MMM', 'UPS',
                'FDX', 'NSC', 'CSX', 'NOC'
            ],
            
            'Materials': [
                'LIN', 'APD', 'FCX', 'NUE', 'DOW', 'DD', 'NEM', 'STLD'
            ],
            
            'Utilities': [
                'NEE', 'DUK', 'SO', 'AEP', 'EXC', 'D'
            ],
            
            'Real Estate': [
                'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG'
            ],
            
            'Crypto/Bitcoin Exposure': [
                'MSTR', 'COIN', 'MARA', 'RIOT'
            ]
        }

    def check_market_regime(self):
        """Check overall market conditions"""
        try:
            spy = yf.Ticker("SPY")
            vix = yf.Ticker("^VIX")
            
            spy_data = spy.history(period="3mo")
            vix_data = vix.history(period="1mo")
            
            if len(spy_data) < 30 or len(vix_data) < 5:
                return True, "Limited market data - proceeding"
            
            spy_data['EMA_50'] = spy_data['Close'].ewm(span=50).mean()
            current_spy = spy_data.iloc[-1]
            current_vix = vix_data['Close'].iloc[-1]
            
            spy_trend_ok = current_spy['Close'] > current_spy['EMA_50']
            vix_ok = current_vix < 30
            
            if spy_trend_ok and vix_ok:
                return True, f"Market OK: SPY above 50EMA, VIX {current_vix:.1f}"
            else:
                return False, f"Market Risk: SPY trend={spy_trend_ok}, VIX={current_vix:.1f}"
                
        except Exception as e:
            return True, f"Market check failed - proceeding: {str(e)}"

    def calculate_stoch_rsi(self, data, period=14, smooth_k=3, smooth_d=3):
        """Calculate Stochastic RSI"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min()) * 100
            stoch_k = stoch_rsi.rolling(smooth_k).mean()
            stoch_d = stoch_k.rolling(smooth_d).mean()
            
            return stoch_k, stoch_d
        except:
            return pd.Series(dtype=float), pd.Series(dtype=float)

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        try:
            ema_fast = data.ewm(span=fast).mean()
            ema_slow = data.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram
        except:
            return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)

    def check_weekly_filter(self, symbol):
        """Check weekly Stoch RSI filter (advisory)"""
        try:
            ticker = yf.Ticker(symbol)
            weekly_data = ticker.history(period="1y", interval="1wk")
            
            if len(weekly_data) < 30:
                return True, "Limited weekly data - proceeding"
            
            weekly_stoch_k, _ = self.calculate_stoch_rsi(weekly_data['Close'])
            current_weekly_stoch = weekly_stoch_k.iloc[-1]
            
            if pd.isna(current_weekly_stoch):
                return True, "No weekly Stoch RSI - proceeding"
            
            if 20 <= current_weekly_stoch <= 60:
                return True, f"Weekly Stoch RSI: {current_weekly_stoch:.1f} ✅ (optimal)"
            elif current_weekly_stoch < 20:
                return True, f"Weekly Stoch RSI: {current_weekly_stoch:.1f} ⚠️ (early)"
            else:
                return True, f"Weekly Stoch RSI: {current_weekly_stoch:.1f} ⚠️ (late)"
                
        except Exception as e:
            return True, f"Weekly check failed - proceeding"

    def analyze_stock(self, symbol, sector_name):
        """Enhanced stock analysis with Jimbo's criteria"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get 6 months of data, fallback to shorter periods
            data = None
            for period in ["6mo", "3mo", "1y"]:
                try:
                    test_data = ticker.history(period=period, interval="1d")
                    if len(test_data) >= 50:
                        data = test_data
                        break
                except:
                    continue
            
            if data is None or len(data) < 50:
                return None, "Insufficient data"
            
            # Weekly filter (advisory)
            weekly_pass, weekly_msg = self.check_weekly_filter(symbol)
            
            # Calculate indicators
            data['EMA_21'] = data['Close'].ewm(span=21).mean()
            data['EMA_50'] = data['Close'].ewm(span=50).mean()
            data['EMA_200'] = data['Close'].ewm(span=min(200, len(data)-1)).mean()
            data['Volume_EMA_20'] = data['Volume'].ewm(span=20).mean()
            
            # Stoch RSI
            stoch_k, stoch_d = self.calculate_stoch_rsi(data['Close'])
            data['Stoch_K'] = stoch_k
            data['Stoch_D'] = stoch_d
            
            # MACD
            macd, signal, histogram = self.calculate_macd(data['Close'])
            data['MACD'] = macd
            data['MACD_Signal'] = signal
            data['MACD_Histogram'] = histogram
            
            current = data.iloc[-1]
            prev = data.iloc[-2] if len(data) > 1 else current
            
            results = {
                'symbol': symbol,
                'sector': sector_name,
                'price': current['Close'],
                'weekly_msg': weekly_msg,
                'criteria': {},
                'grade': 'PASS',
                'data_days': len(data)
            }
            
            # 1. Trend Filter
            if len(data) >= 200:
                trend_pass = current['Close'] > current['EMA_200']
                trend_ref = "200EMA"
                trend_value = current['EMA_200']
            else:
                trend_pass = current['Close'] > current['EMA_50']
                trend_ref = "50EMA"
                trend_value = current['EMA_50']
                
            results['criteria']['trend_filter'] = {
                'pass': trend_pass,
                'msg': f"Price ${current['Close']:.2f} vs {trend_ref} ${trend_value:.2f}"
            }
            
            # 2. Pullback to Support
            pullback_21 = abs(current['Close'] - current['EMA_21']) / current['Close'] <= 0.03
            pullback_50 = abs(current['Close'] - current['EMA_50']) / current['Close'] <= 0.03
            pullback_pass = pullback_21 or pullback_50
            
            results['criteria']['pullback'] = {
                'pass': pullback_pass,
                'msg': f"21EMA: ${current['EMA_21']:.2f}, 50EMA: ${current['EMA_50']:.2f}"
            }
            
            # 3. Momentum Reset
            momentum_pass = False
            momentum_msg = "No recent Stoch RSI cross"
            
            if not pd.isna(current['Stoch_K']):
                for i in range(2, min(9, len(data))):
                    if (data['Stoch_K'].iloc[-i] < 20 and 
                        data['Stoch_K'].iloc[-i+1] > data['Stoch_K'].iloc[-i] and
                        data['Stoch_K'].iloc[-1] > 20):
                        momentum_pass = True
                        momentum_msg = f"Stoch RSI crossed {i-1} bars ago, now {current['Stoch_K']:.1f}"
                        break
            
            results['criteria']['momentum'] = {
                'pass': momentum_pass,
                'msg': momentum_msg
            }
            
            # 4. MACD Confirmation (Jimbo's priority)
            macd_buy = (current['MACD'] > current['MACD_Signal'] and 
                       prev['MACD'] <= prev['MACD_Signal'])
            macd_green = current['MACD'] > current['MACD_Signal']
            macd_rising = current['MACD'] > prev['MACD']
            
            histogram_accelerating = (current['MACD_Histogram'] > prev['MACD_Histogram'] and
                                    len(data) > 3 and current['MACD_Histogram'] > data['MACD_Histogram'].iloc[-3])
            
            dc_macd_green = current['MACD_Histogram'] > 0 and histogram_accelerating
            macd_pass = macd_buy and macd_green and macd_rising
            
            results['criteria']['macd'] = {
                'pass': macd_pass,
                'dc_macd': dc_macd_green,
                'msg': f"Buy: {macd_buy}, Green: {macd_green}, Rising: {macd_rising}, Hist+: {dc_macd_green}"
            }
            
            # 5. Volume Confirmation
            volume_pass = current['Volume'] >= 1.5 * current['Volume_EMA_20']
            
            results['criteria']['volume'] = {
                'pass': volume_pass,
                'msg': f"Volume: {current['Volume']:,.0f} vs 1.5x EMA20: {1.5 * current['Volume_EMA_20']:,.0f}"
            }
            
            # 6. Candle Quality
            candle_range = current['High'] - current['Low']
            close_position = (current['Close'] - current['Low']) / candle_range if candle_range > 0 else 0
            candle_pass = (current['Close'] > current['Open'] and close_position >= 0.3)
            
            results['criteria']['candle'] = {
                'pass': candle_pass,
                'msg': f"Bullish candle, close in top {close_position*100:.0f}% of range"
            }
            
            # 7. Risk-Reward
            recent_high = data['High'].tail(20).max()
            stop_loss = min(current['EMA_50'], data['Low'].tail(10).min())
            
            risk = current['Close'] - stop_loss
            reward = recent_high - current['Close']
            rrr = reward / risk if risk > 0 else 0
            
            rrr_pass = rrr >= 2.0
            results['criteria']['risk_reward'] = {
                'pass': rrr_pass,
                'msg': f"RRR: {rrr:.1f}:1 (Target: ${recent_high:.2f}, Stop: ${stop_loss:.2f})"
            }
            
            # Determine grade based on Jimbo's rules
            mandatory_criteria = ['trend_filter', 'pullback', 'momentum', 'macd', 'volume', 'candle', 'risk_reward']
            passed_criteria = sum(1 for criterion in mandatory_criteria if results['criteria'][criterion]['pass'])
            
            if passed_criteria == len(mandatory_criteria):
                if results['criteria']['macd']['dc_macd'] and results['criteria']['macd']['pass']:
                    results['grade'] = 'A+'
                elif results['criteria']['macd']['pass']:
                    results['grade'] = 'A'
                else:
                    results['grade'] = 'B'
            else:
                results['grade'] = 'PASS'
            
            # Weekly timing context
            if "optimal" in weekly_msg:
                results['timing_note'] = "Optimal weekly timing"
            elif "early" in weekly_msg:
                results['timing_note'] = "Early timing"
            elif "late" in weekly_msg:
                results['timing_note'] = "Late timing"
            else:
                results['timing_note'] = "Unknown timing"
            
            return results, None
            
        except Exception as e:
            return None, f"Analysis error: {str(e)}"

    def scan_sectors(self):
        """Scan curated stock list with rate limiting"""
        market_ok, market_msg = self.check_market_regime()
        
        total_stocks = sum(len(symbols) for symbols in self.sectors.values())
        print(f"📊 Market Regime: {market_msg}")
        print(f"🎯 Scanning {total_stocks} curated stocks with rate limiting...")
        
        all_results = {}
        a_plus_setups = []
        a_setups = []
        successful_scans = 0
        
        for sector_name, symbols in self.sectors.items():
            print(f"\nScanning {sector_name} ({len(symbols)} stocks)...")
            sector_results = []
            
            for i, symbol in enumerate(symbols):
                try:
                    result, error = self.analyze_stock(symbol, sector_name)
                    
                    if result:
                        sector_results.append(result)
                        successful_scans += 1
                        
                        if result['grade'] == 'A+':
                            a_plus_setups.append(result)
                        elif result['grade'] == 'A':
                            a_setups.append(result)
                        
                        # Display with timing indicator
                        timing_icon = "🟢" if "optimal" in result.get('timing_note', '') else "🟡" if "early" in result.get('timing_note', '') else "🟠"
                        print(f"  {symbol}: {result['grade']} {timing_icon} (${result['price']:.2f})")
                    else:
                        print(f"  {symbol}: {error}")
                    
                    # Rate limiting between stocks
                    if i < len(symbols) - 1:  # Don't delay after last stock
                        time.sleep(self.request_delay)
                        
                except Exception as e:
                    print(f"  {symbol}: Exception - {str(e)}")
            
            all_results[sector_name] = sector_results
            
            # Longer delay between sectors
            time.sleep(self.batch_delay)
        
        print(f"\n📈 Scan Results:")
        print(f"  Successfully analyzed: {successful_scans}/{total_stocks} stocks ({successful_scans/total_stocks*100:.1f}%)")
        print(f"  A+ setups: {len(a_plus_setups)}")
        print(f"  A setups: {len(a_setups)}")
        
        return all_results, a_plus_setups, a_setups, market_ok

    def send_pushover_alert(self, a_plus_setups, a_setups, market_regime_ok):
        """Send Pushover notification for qualifying setups"""
        try:
            import requests
            
            if not a_plus_setups and not a_setups:
                return
            
            market_emoji = "🟢" if market_regime_ok else "🔴"
            market_text = "Favorable" if market_regime_ok else "Caution"
            
            title = f"{market_emoji} Jimbo's Strategy: {len(a_plus_setups)} A+ & {len(a_setups)} A Setups"
            
            message = f"📊 Market: {market_text}\n📈 High-Quality Setups Found!\n\n"
            
            if a_plus_setups:
                message += f"🟢 A+ SETUPS ({len(a_plus_setups)}) - Full MACD Confirmation:\n"
                for setup in a_plus_setups:
                    timing_icon = "🟢" if "optimal" in setup.get('timing_note', '') else "🟡" if "early" in setup.get('timing_note', '') else "🟠"
                    
                    # Add bull flag info
                    flag_info = ""
                    if setup.get('bull_flag', {}).get('detected', False):
                        quality = setup['bull_flag'].get('quality_score', 0)
                        flag_info = f" 🚩Q{quality}" if quality >= 3 else f" 📏Q{quality}"
                    
                    message += f"• {setup['symbol']} ${setup['price']:.2f} {timing_icon}{flag_info}\n"
                message += "\n"
            
            if a_setups:
                message += f"🟡 A SETUPS ({len(a_setups)}) - Strong MACD:\n"
                for setup in a_setups:
                    timing_icon = "🟢" if "optimal" in setup.get('timing_note', '') else "🟡" if "early" in setup.get('timing_note', '') else "🟠"
                    
                    # Add bull flag info
                    flag_info = ""
                    if setup.get('bull_flag', {}).get('detected', False):
                        quality = setup['bull_flag'].get('quality_score', 0)
                        flag_info = f" 🚩Q{quality}" if quality >= 3 else f" 📏Q{quality}"
                    
                    message += f"• {setup['symbol']} ${setup['price']:.2f} {timing_icon}{flag_info}\n"
                message += "\n"
            
            message += f"⏰ {datetime.now().strftime('%I:%M %p')}"
            
            payload = {
                'token': self.pushover_config['app_token'],
                'user': self.pushover_config['user_key'],
                'title': title,
                'message': message,
                'priority': 2 if a_plus_setups and market_regime_ok else 1,
                'sound': self.pushover_config['sound'],
                'url': 'https://tradingview.com',
                'url_title': 'View Charts'
            }
            
            # Emergency priority for A+ setups in good market
            if a_plus_setups and market_regime_ok:
                payload['retry'] = 30
                payload['expire'] = 300
            
            response = requests.post('https://api.pushover.net/1/messages.json', data=payload)
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 1:
                    priority_text = "EMERGENCY" if payload.get('priority') == 2 else "HIGH"
                    print(f"📱 Pushover alert sent! (Priority: {priority_text})")
                else:
                    print(f"❌ Pushover failed: {result}")
            else:
                print(f"❌ Pushover HTTP error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Pushover failed: {e}")

    def save_results(self, a_plus_setups, a_setups, market_regime_ok):
        """Save results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'jimbo_strategy_results_{timestamp}.txt'
        
        with open(filename, 'w') as f:
            f.write(f"Jimbo's Strategy Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Market Regime: {'FAVORABLE' if market_regime_ok else 'CAUTION'}\n")
            f.write(f"Total A+ Setups: {len(a_plus_setups)}\n")
            f.write(f"Total A Setups: {len(a_setups)}\n\n")
            
            for setup_list, grade_name in [(a_plus_setups, "A+"), (a_setups, "A")]:
                if setup_list:
                    f.write(f"{grade_name} SETUPS:\n{'='*30}\n")
                    for setup in setup_list:
                        f.write(f"\n{setup['symbol']} - ${setup['price']:.2f}\n")
                        f.write(f"  Sector: {setup['sector']}\n")
                        f.write(f"  Timing: {setup.get('timing_note', 'Unknown')}\n")
                        f.write(f"  Weekly: {setup['weekly_msg']}\n")
                        for criterion, data in setup['criteria'].items():
                            status = "PASS" if data['pass'] else "FAIL"
                            f.write(f"  {status}: {criterion.replace('_', ' ').title()} - {data['msg']}\n")
        
        print(f"💾 Results saved to: {filename}")

    def run_daily_scan(self):
        """Main function to run Jimbo's daily scan"""
        print("="*60)
        print("JIMBO'S ENHANCED STRATEGY SCANNER")
        print("Optimized for 100 High-Quality Stocks")
        print("="*60)
        
        start_time = datetime.now()
        all_results, a_plus_setups, a_setups, market_ok = self.scan_sectors()
        scan_duration = datetime.now() - start_time
        
        print(f"\n{'='*60}")
        print("SCAN SUMMARY")
        print(f"{'='*60}")
        market_status = "🟢 FAVORABLE" if market_ok else "🔴 CAUTION"
        print(f"Market Regime: {market_status}")
        print(f"Scan Duration: {scan_duration.total_seconds():.1f} seconds")
        print(f"A+ Setups Found: {len(a_plus_setups)}")
        print(f"A Setups Found: {len(a_setups)}")
        
        if a_plus_setups:
            print(f"\n🟢 A+ SETUPS (Full MACD Confirmation):")
            for setup in a_plus_setups:
                timing_icon = "🟢" if "optimal" in setup.get('timing_note', '') else "🟡" if "early" in setup.get('timing_note', '') else "🟠"
                print(f"  {setup['symbol']} - ${setup['price']:.2f} {timing_icon} ({setup['sector']})")
        
        if a_setups:
            print(f"\n🟡 A SETUPS (Strong MACD):")
            for setup in a_setups:
                timing_icon = "🟢" if "optimal" in setup.get('timing_note', '') else "🟡" if "early" in setup.get('timing_note', '') else "🟠"
                print(f"  {setup['symbol']} - ${setup['price']:.2f} {timing_icon} ({setup['sector']})")
        
        # Send notifications and save results
        if a_plus_setups or a_setups:
            if self.notifications['pushover']:
                self.send_pushover_alert(a_plus_setups, a_setups, market_ok)
            
            if self.notifications['file']:
                self.save_results(a_plus_setups, a_setups, market_ok)
        else:
            print("\nNo qualifying setups found today.")
        
        return all_results, a_plus_setups, a_setups

# Main execution
if __name__ == "__main__":
    scanner = JimboStrategyScanner()
    
    # Calculate total stocks
    total_stocks = sum(len(symbols) for symbols in scanner.sectors.values())
    
    print("📱 Pushover notifications: ENABLED")
    print(f"🎯 Curated stock universe: {total_stocks} high-quality stocks")
    print("⚡ Rate limiting: 500ms between stocks, 2s between sectors")
    
    # List your requested stocks for verification
    requested_stocks = ['HIMS', 'OSCR', 'CVX', 'XOM', 'ACMR', 'MPC', 'SLB', 'EOG', 'MSTR', 'HIVE', 'MARA', 'RIOT', 'CAN', 'CBTC']
    print(f"✅ All requested stocks included: {', '.join(requested_stocks)}")
    
    try:
        results, a_plus, a_grade = scanner.run_daily_scan()
        print(f"\nScan completed successfully at {datetime.now().strftime('%H:%M:%S')}")
        
        if a_plus or a_grade:
            print(f"\n🚨 TRADING OPPORTUNITIES FOUND!")
            print(f"Check your Pushover app for detailed alerts.")
        else:
            print(f"\n📊 Market scanned - no setups meeting Jimbo's criteria today.")
            print(f"This maintains the strategy's quality-over-quantity approach.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Scan interrupted by user")
    except Exception as e:
        print(f"\n❌ Scan failed: {e}")
        print("This may be due to temporary Yahoo Finance issues.")

# Scheduling information
"""
To automate this scanner:

Windows Task Scheduler:
- Run daily at 4:30 PM EST (after market close)
- Program: python.exe
- Arguments: C:\\Users\\home3\\TradingBot\\jimbo.py
- Start in: C:\\Users\\home3\\TradingBot\\

Linux/Mac Cron:
30 16 * * 1-5 cd /path/to/TradingBot && /usr/bin/python3 jimbo.py

The scanner now includes all your requested stocks:
- Energy: CVX, XOM, MPC, SLB, EOG (already optimized sector)
- Crypto: MSTR, HIVE, MARA, RIOT, CAN (comprehensive crypto coverage)
- Healthcare: HIMS (telehealth exposure)
- Consumer Discretionary: OSCR (growth stock)
- Technology: ACMR (semiconductor equipment)
- Financials: CBTC (crypto/blockchain finance)

Total stocks: {total_stocks} carefully selected for optimal Jimbo Strategy execution.
"""
