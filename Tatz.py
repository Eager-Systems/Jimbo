#!/usr/bin/env python3
"""
Pre-Breakout Scanner - Monthly Timeframe Alert System
Catches stocks BEFORE they breakout like HOOD did in 2023
WORKING VERSION - All functions included with Pushover notifications
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import sys
import argparse
import requests
import json
import os
warnings.filterwarnings('ignore')

def load_pushover_config():
    """Load Pushover configuration - credentials hardcoded for simplicity"""
    # Your Pushover credentials
    app_token = "aejpnud7k86k5f2kxeycfxr4uram5g"
    user_key = "u9uzeesuxqax45yhjrzyfvv6opbm6y"
    
    return app_token, user_key

def send_pushover_alert(app_token, user_key, title, message, priority=0):
    """Send Pushover notification"""
    if not app_token or not user_key:
        return False
    
    try:
        data = {
            'token': app_token,
            'user': user_key,
            'title': title,
            'message': message,
            'priority': priority
        }
        
        response = requests.post('https://api.pushover.net/1/messages.json', data=data, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Pushover error: {e}")
        return False

class PreBreakoutScanner:
    def __init__(self):
        self.monthly_signals = []
        print("🎯 Pre-Breakout Scanner Initialized")
        print("Looking for stocks BEFORE they make big moves...")
        
    def analyze_monthly_setup(self, symbol):
        """Analyze monthly timeframe for early breakout signals"""
        try:
            # Get 5 years of data for proper monthly analysis
            ticker = yf.Ticker(symbol)
            daily_data = ticker.history(period="5y")
            
            if daily_data.empty or len(daily_data) < 500:
                return None
            
            # Create monthly data
            monthly_data = self.create_monthly_data(daily_data)
            
            if len(monthly_data) < 24:  # Need at least 2 years of monthly data
                return None
            
            # Analyze the setup
            setup_analysis = self.detect_pre_breakout_setup(monthly_data, symbol)
            
            if setup_analysis and setup_analysis['alert_score'] >= 60:  # Lower threshold for more results
                return setup_analysis
                
            return None
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None
    
    def create_monthly_data(self, daily_data):
        """Convert daily data to monthly bars"""
        # Resample to monthly - use last business day of month
        monthly = daily_data.resample('M').agg({
            'Open': 'first',
            'High': 'max', 
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return monthly
    
    def detect_pre_breakout_setup(self, monthly_data, symbol):
        """Detect pre-breakout conditions on monthly timeframe"""
        
        # Key metrics for early detection
        current_price = monthly_data['Close'].iloc[-1]
        current_month = monthly_data.iloc[-1]
        
        # 1. MULTI-YEAR BASE ANALYSIS
        base_analysis = self.analyze_multi_year_base(monthly_data)
        
        # 2. VOLUME PATTERN ANALYSIS  
        volume_analysis = self.analyze_volume_accumulation(monthly_data)
        
        # 3. MONTHLY TECHNICAL SETUP
        technical_analysis = self.analyze_monthly_technicals(monthly_data)
        
        # 4. RELATIVE STRENGTH ANALYSIS
        strength_analysis = self.analyze_relative_strength(monthly_data)
        
        # 5. EARLY BREAKOUT SIGNALS
        breakout_signals = self.detect_early_breakout_signals(monthly_data)
        
        # Calculate overall alert score
        alert_score = self.calculate_alert_score(
            base_analysis, volume_analysis, technical_analysis,
            strength_analysis, breakout_signals
        )
        
        return {
            'symbol': symbol,
            'alert_score': alert_score,
            'current_price': current_price,
            'base_analysis': base_analysis,
            'volume_analysis': volume_analysis,
            'technical_analysis': technical_analysis,
            'strength_analysis': strength_analysis,
            'breakout_signals': breakout_signals,
            'monthly_action': self.get_monthly_action(alert_score, breakout_signals),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def analyze_multi_year_base(self, monthly_data):
        """Analyze multi-year base formation like HOOD 2021-2024"""
        
        # Find all-time high and when it occurred
        ath_price = monthly_data['High'].max()
        ath_idx = monthly_data['High'].idxmax()
        current_price = monthly_data['Close'].iloc[-1]
        
        # Calculate decline from ATH
        decline_from_ath = (current_price - ath_price) / ath_price * 100
        
        # Months since ATH
        months_since_ath = len(monthly_data) - monthly_data.index.get_loc(ath_idx) - 1
        
        # Analyze base formation (look for sideways action)
        if months_since_ath >= 6:  # At least 6 months since ATH
            # Get data since ATH
            base_data = monthly_data.loc[ath_idx:]
            
            # Calculate base characteristics
            base_high = base_data['High'].max()
            base_low = base_data['Low'].min()
            base_range = (base_high - base_low) / base_data['Close'].mean() * 100
            
            # Look for tightening range (sign of accumulation)
            recent_12m = base_data.tail(min(12, len(base_data)))
            recent_range = (recent_12m['High'].max() - recent_12m['Low'].min()) / recent_12m['Close'].mean() * 100
            
            # Base quality signals
            is_tightening = recent_range < base_range * 0.8 if base_range > 0 else False
            sufficient_time = months_since_ath >= 12  # At least 12 months
            proper_decline = -90 <= decline_from_ath <= -20  # Proper reset
            
            base_score = 0
            if proper_decline: base_score += 25
            if sufficient_time: base_score += 20  
            if is_tightening: base_score += 20
            if months_since_ath >= 24: base_score += 15  # Long base bonus
            if -80 <= decline_from_ath <= -40: base_score += 15  # Sweet spot
            if months_since_ath >= 36: base_score += 5   # Very long base
            
            return {
                'base_detected': True,
                'base_score': min(base_score, 100),
                'months_since_ath': months_since_ath,
                'decline_from_ath': decline_from_ath,
                'base_range': base_range,
                'recent_range': recent_range,
                'is_tightening': is_tightening
            }
        
        return {
            'base_detected': False,
            'base_score': 0,
            'months_since_ath': months_since_ath,
            'decline_from_ath': decline_from_ath
        }
    
    def analyze_volume_accumulation(self, monthly_data):
        """Detect volume accumulation patterns"""
        
        if len(monthly_data) < 12:
            return {'volume_score': 0, 'volume_trend_up': False, 'smart_money_signal': False, 'volume_ratio': 1.0}
        
        # Volume trend analysis
        volume_ma_6 = monthly_data['Volume'].rolling(6).mean()
        volume_ma_12 = monthly_data['Volume'].rolling(12).mean()
        
        current_volume = monthly_data['Volume'].iloc[-1]
        recent_avg_volume = monthly_data['Volume'].tail(3).mean()
        historical_avg = monthly_data['Volume'].mean()
        
        # Accumulation signals
        volume_trend_up = volume_ma_6.iloc[-1] > volume_ma_12.iloc[-1] if len(volume_ma_12.dropna()) > 0 else False
        volume_pickup = recent_avg_volume > historical_avg * 1.05
        current_above_avg = current_volume > volume_ma_6.iloc[-1] if not pd.isna(volume_ma_6.iloc[-1]) else False
        
        # Smart money accumulation (volume on up months vs down months)
        monthly_returns = monthly_data['Close'].pct_change().dropna()
        up_months = monthly_returns > 0
        down_months = monthly_returns < 0
        
        if up_months.sum() > 2 and down_months.sum() > 2:
            avg_volume_up = monthly_data.loc[monthly_returns[up_months].index, 'Volume'].mean()
            avg_volume_down = monthly_data.loc[monthly_returns[down_months].index, 'Volume'].mean()
            smart_money_signal = avg_volume_up > avg_volume_down * 1.1
        else:
            smart_money_signal = False
        
        volume_score = 0
        if volume_trend_up: volume_score += 15
        if volume_pickup: volume_score += 10
        if current_above_avg: volume_score += 10
        if smart_money_signal: volume_score += 20
        
        # Recent volume surge
        if recent_avg_volume > historical_avg * 1.3: volume_score += 15
        
        return {
            'volume_score': min(volume_score, 100),
            'volume_trend_up': volume_trend_up,
            'volume_pickup': volume_pickup,
            'smart_money_signal': smart_money_signal,
            'volume_ratio': recent_avg_volume / historical_avg if historical_avg > 0 else 1.0
        }
    
    def analyze_monthly_technicals(self, monthly_data):
        """Analyze monthly technical indicators"""
        
        if len(monthly_data) < 20:
            return {'technical_score': 0}
        
        # Monthly moving averages
        ma_6 = monthly_data['Close'].rolling(6).mean()
        ma_12 = monthly_data['Close'].rolling(12).mean()
        ma_24 = monthly_data['Close'].rolling(24).mean()
        
        current_price = monthly_data['Close'].iloc[-1]
        
        # Monthly RSI
        monthly_rsi = self.calculate_rsi(monthly_data['Close'], 14)
        current_rsi = monthly_rsi.iloc[-1] if len(monthly_rsi.dropna()) > 0 else 50
        
        # Moving average alignment and position
        above_ma6 = current_price > ma_6.iloc[-1] if not pd.isna(ma_6.iloc[-1]) else False
        above_ma12 = current_price > ma_12.iloc[-1] if not pd.isna(ma_12.iloc[-1]) else False
        above_ma24 = current_price > ma_24.iloc[-1] if not pd.isna(ma_24.iloc[-1]) else False
        
        # MA slope (momentum)
        ma6_slope = ((ma_6.iloc[-1] - ma_6.iloc[-3]) / ma_6.iloc[-3] * 100) if len(ma_6.dropna()) >= 3 and ma_6.iloc[-3] > 0 else 0
        ma12_slope = ((ma_12.iloc[-1] - ma_12.iloc[-6]) / ma_12.iloc[-6] * 100) if len(ma_12.dropna()) >= 6 and ma_12.iloc[-6] > 0 else 0
        
        # Monthly momentum
        monthly_momentum = self.calculate_monthly_momentum(monthly_data)
        
        technical_score = 0
        if above_ma6: technical_score += 10
        if above_ma12: technical_score += 15  
        if above_ma24: technical_score += 20  # Most important for monthly
        if ma6_slope > 0: technical_score += 10
        if ma12_slope > 0: technical_score += 15
        if 35 <= current_rsi <= 75: technical_score += 15  # Wider range for monthly
        if monthly_momentum > 0: technical_score += 15
        
        return {
            'technical_score': min(technical_score, 100),
            'above_ma6': above_ma6,
            'above_ma12': above_ma12, 
            'above_ma24': above_ma24,
            'monthly_rsi': current_rsi,
            'ma6_slope': ma6_slope,
            'ma12_slope': ma12_slope,
            'monthly_momentum': monthly_momentum
        }
    
    def analyze_relative_strength(self, monthly_data):
        """Analyze relative strength vs market"""
        try:
            # Get SPY data for comparison
            spy = yf.Ticker("SPY")
            spy_data = spy.history(period="5y")
            spy_monthly = spy_data.resample('M').agg({'Close': 'last'}).dropna()
            
            # Align dates
            common_dates = monthly_data.index.intersection(spy_monthly.index)
            if len(common_dates) < 6:
                return {'strength_score': 50, 'relative_strength': 0, 'rs_trend': 0}
            
            stock_aligned = monthly_data.loc[common_dates, 'Close']
            spy_aligned = spy_monthly.loc[common_dates, 'Close']
            
            # Calculate relative strength (6-month performance)
            periods_back = min(6, len(stock_aligned)-1)
            if periods_back <= 0:
                return {'strength_score': 50, 'relative_strength': 0, 'rs_trend': 0}
            
            stock_return = (stock_aligned.iloc[-1] / stock_aligned.iloc[-periods_back-1] - 1) if stock_aligned.iloc[-periods_back-1] > 0 else 0
            spy_return = (spy_aligned.iloc[-1] / spy_aligned.iloc[-periods_back-1] - 1) if spy_aligned.iloc[-periods_back-1] > 0 else 0
            
            relative_strength = stock_return - spy_return
            
            # Trend of relative strength (3-month vs 6-month)
            periods_mid = min(3, len(stock_aligned)-1)
            if periods_mid > 0:
                stock_return_3m = (stock_aligned.iloc[-1] / stock_aligned.iloc[-periods_mid-1] - 1) if stock_aligned.iloc[-periods_mid-1] > 0 else 0
                spy_return_3m = (spy_aligned.iloc[-1] / spy_aligned.iloc[-periods_mid-1] - 1) if spy_aligned.iloc[-periods_mid-1] > 0 else 0
                rs_trend = (stock_return_3m - spy_return_3m) - (stock_return - spy_return)
            else:
                rs_trend = 0
            
            strength_score = 50  # Neutral base
            if relative_strength > 0.15: strength_score += 25  # Outperforming by 15%+
            elif relative_strength > 0.05: strength_score += 15
            elif relative_strength > -0.05: strength_score += 5
            
            if rs_trend > 0: strength_score += 10  # Improving relative strength
            
            return {
                'strength_score': min(strength_score, 100),
                'relative_strength': relative_strength,
                'rs_trend': rs_trend
            }
            
        except Exception as e:
            return {'strength_score': 50, 'relative_strength': 0, 'rs_trend': 0}
    
    def detect_early_breakout_signals(self, monthly_data):
        """Detect early signals that breakout is imminent"""
        
        if len(monthly_data) < 2:
            return {'signal_score': 0, 'monthly_breakout': False, 'highest_close_12m': False, 
                   'volume_expansion': False, 'range_expansion': False, 'breaking_resistance': False}
        
        current_month = monthly_data.iloc[-1]
        previous_month = monthly_data.iloc[-2]
        
        # 1. Monthly close above previous month's high
        monthly_breakout = current_month['Close'] > previous_month['High']
        
        # 2. Highest close in X months
        lookback_months = min(12, len(monthly_data))
        recent_highs = monthly_data['Close'].tail(lookback_months)
        highest_close_12m = current_month['Close'] == recent_highs.max()
        
        # 3. Volume expansion on current month
        if len(monthly_data) >= 6:
            avg_volume = monthly_data['Volume'].rolling(6).mean().iloc[-1]
            volume_expansion = current_month['Volume'] > avg_volume * 1.2 if not pd.isna(avg_volume) and avg_volume > 0 else False
        else:
            volume_expansion = False
        
        # 4. Monthly range expansion (volatility pickup)
        current_range = (current_month['High'] - current_month['Low']) / current_month['Close'] if current_month['Close'] > 0 else 0
        if len(monthly_data) >= 6:
            avg_range = ((monthly_data['High'] - monthly_data['Low']) / monthly_data['Close']).rolling(6).mean().iloc[-1]
            range_expansion = current_range > avg_range * 1.15 if not pd.isna(avg_range) and avg_range > 0 else False
        else:
            range_expansion = False
        
        # 5. Breaking key resistance levels
        if len(monthly_data) >= 12:
            resistance_level = monthly_data['High'].rolling(12).max().iloc[-2]  # 12-month high (excluding current)
            breaking_resistance = current_month['Close'] > resistance_level * 0.95 if not pd.isna(resistance_level) else False
        else:
            breaking_resistance = False
        
        signal_score = 0
        if monthly_breakout: signal_score += 25
        if highest_close_12m: signal_score += 20
        if volume_expansion: signal_score += 20
        if range_expansion: signal_score += 10
        if breaking_resistance: signal_score += 25
        
        return {
            'signal_score': min(signal_score, 100),
            'monthly_breakout': monthly_breakout,
            'highest_close_12m': highest_close_12m,
            'volume_expansion': volume_expansion,
            'range_expansion': range_expansion,
            'breaking_resistance': breaking_resistance
        }
    
    def calculate_alert_score(self, base_analysis, volume_analysis, technical_analysis, strength_analysis, breakout_signals):
        """Calculate overall alert score (0-100)"""
        
        # Weighted scoring - emphasize base formation and breakout signals
        base_weight = 0.35      # Most important - proper base formation
        signal_weight = 0.25    # Early breakout signals
        technical_weight = 0.20 # Monthly technical setup
        volume_weight = 0.15    # Volume accumulation  
        strength_weight = 0.05  # Relative strength (nice to have)
        
        total_score = (
            base_analysis['base_score'] * base_weight +
            breakout_signals['signal_score'] * signal_weight +
            technical_analysis['technical_score'] * technical_weight +
            volume_analysis['volume_score'] * volume_weight + 
            strength_analysis['strength_score'] * strength_weight / 100 * 100
        )
        
        return min(total_score, 100)
    
    def get_monthly_action(self, alert_score, breakout_signals):
        """Determine monthly timeframe action"""
        
        if alert_score >= 85 and breakout_signals['monthly_breakout']:
            return "🚨 STRONG BUY - Monthly breakout confirmed"
        elif alert_score >= 80:
            return "🔥 BUY - Excellent monthly setup"
        elif alert_score >= 70:
            return "📈 ACCUMULATE - Strong monthly base"
        elif alert_score >= 60:
            return "👀 WATCH - Developing setup"
        else:
            return "⏳ WAIT - Setup not ready"
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return pd.Series([50] * len(prices), index=prices.index)
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # Handle division by zero
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_monthly_momentum(self, monthly_data):
        """Calculate monthly momentum score"""
        if len(monthly_data) < 7:
            return 0
        
        # 3-month vs 6-month performance
        current_price = monthly_data['Close'].iloc[-1]
        price_3m_ago = monthly_data['Close'].iloc[-4] if len(monthly_data) >= 4 else current_price
        price_6m_ago = monthly_data['Close'].iloc[-7] if len(monthly_data) >= 7 else current_price
        
        if price_3m_ago > 0 and price_6m_ago > 0:
            momentum_3m = (current_price - price_3m_ago) / price_3m_ago * 100
            momentum_6m = (current_price - price_6m_ago) / price_6m_ago * 100
            
            # Acceleration (3m momentum > 6m momentum = accelerating)
            return momentum_3m - (momentum_6m / 2)  # Weight recent momentum more
        
        return 0
    
    def scan_universe_for_monthly_setups(self, stock_universe):
        """Scan entire universe for monthly buy signals"""
        
        print(f"🎯 Scanning {len(stock_universe)} stocks for Monthly Buy Signals...")
        print("Looking for stocks BEFORE they breakout like HOOD...")
        print("=" * 60)
        
        alerts = []
        processed = 0
        
        for symbol in stock_universe:
            try:
                processed += 1
                print(f"[{processed:3d}/{len(stock_universe)}] {symbol:6}", end=" ")
                
                setup = self.analyze_monthly_setup(symbol)
                
                if setup:
                    alerts.append(setup)
                    print(f"ALERT! Score: {setup['alert_score']:.0f}")
                else:
                    print("No setup")
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Sort by alert score
        alerts.sort(key=lambda x: x['alert_score'], reverse=True)
        
        return alerts
    
    def display_monthly_alerts(self, alerts):
        """Display monthly buy alerts"""
        
        if not alerts:
            print("\n❌ No monthly buy signals found")
            print("💡 Try running again later or adjust criteria")
            return
        
        print(f"\n🚨 MONTHLY BUY ALERTS ({len(alerts)} found)")
        print("=" * 80)
        
        for i, alert in enumerate(alerts, 1):
            print(f"\n{i:2d}. {alert['symbol']} - Score: {alert['alert_score']:.0f}/100")
            print(f"    Action: {alert['monthly_action']}")
            print(f"    Price: ${alert['current_price']:.2f}")
            
            base = alert['base_analysis']
            if base['base_detected']:
                print(f"    Base: {base['months_since_ath']} months, {base['decline_from_ath']:.1f}% from ATH")
            else:
                print(f"    Base: {base['months_since_ath']} months since ATH, {base['decline_from_ath']:.1f}% decline")
            
            volume = alert['volume_analysis'] 
            print(f"    Volume: {volume['volume_ratio']:.1f}x avg, Smart Money: {volume['smart_money_signal']}")
            
            # Display bull flag information if detected
            bull_flag = alert['bull_flag_analysis']
            if bull_flag['flag_detected']:
                print(f"    Bull Flag: Score {bull_flag['flag_score']:.0f}, Pole Strength: {bull_flag['flagpole_strength']:.0f}")
                if bull_flag.get('breakout_imminent', False):
                    print(f"    🚨 FLAG BREAKOUT IMMINENT!")
            
            signals = alert['breakout_signals']
            signal_indicators = []
            if signals['monthly_breakout']:
                signal_indicators.append("🚨 MONTHLY BREAKOUT")
            if signals['highest_close_12m']:
                signal_indicators.append("📈 12M HIGH")
            if signals['volume_expansion']:
                signal_indicators.append("📊 VOLUME SURGE")
            if signals['breaking_resistance']:
                signal_indicators.append("🔥 RESISTANCE BREAK")
                
            if signal_indicators:
                print(f"    Signals: {' | '.join(signal_indicators)}")

def get_comprehensive_universe():
    """Get comprehensive universe for monthly scanning"""
    
    # All sectors - comprehensive coverage prioritizing likely candidates
    universe = [
        # High Priority: Crypto ecosystem (likely to have HOOD-like patterns)
        'COIN', 'HOOD', 'RIOT', 'MARA', 'CLSK', 'BITF', 'HUT', 'MSTR', 'CORZ',
        
        # High Priority: Recent IPOs and growth stocks
        'SOFI', 'RBLX', 'PLTR', 'AFRM', 'UPST', 'PATH', 'DOCN', 'SNOW',
        'CRWD', 'ZS', 'NET', 'DDOG', 'OKTA', 'TWLO', 'DOCU', 'TEAM',
        
        # Medium Priority: Beaten down tech
        'SQ', 'PYPL', 'SHOP', 'ROKU', 'PINS', 'SNAP', 'PTON', 'ZM',
        'UBER', 'LYFT', 'SPOT', 'NFLX', 'META',
        
        # Medium Priority: Clean Energy
        'ENPH', 'SEDG', 'FSLR', 'RUN', 'CHPT', 'BLNK', 'TSLA',
        
        # Lower Priority: Traditional Value (for next rotation)
        'CAT', 'DE', 'HON', 'ITW', 'GE', 'BA', 'RTX', 'LMT',
        'JPM', 'BAC', 'WFC', 'RF', 'KEY', 'CFG',
        'XOM', 'CVX', 'COP', 'MRO', 'DVN', 'EOG',
        
        # Healthcare/Biotech
        'JNJ', 'PFE', 'MRNA', 'BNTX', 'NVAX', 'SGEN', 'BEAM', 'CRSP',
        
        # Semiconductors  
        'NVDA', 'AMD', 'INTC', 'QCOM', 'AVGO', 'MU', 'MRVL',
        
        # Additional recent IPOs and SPACs
        'BILL', 'FIVN', 'GTLB', 'BROS', 'LCID', 'RIVN', 'OPEN', 'CLOV'
    ]
    
    return universe

def get_focused_universe():
    """Get focused universe of most likely candidates"""
    return [
        # Crypto ecosystem - highest probability
        'COIN', 'HOOD', 'RIOT', 'MARA', 'SOFI', 'CLSK', 'MSTR',
        
        # Recent IPOs - second highest  
        'RBLX', 'PLTR', 'AFRM', 'UPST', 'PATH', 'CRWD', 'SNOW',
        
        # Beaten down growth
        'SQ', 'ROKU', 'SHOP', 'PINS', 'PTON', 'ENPH', 'SEDG'
    ]

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Pre-Breakout Scanner - Monthly Timeframe')
    parser.add_argument('--focused', action='store_true', help='Scan focused universe (faster)')
    parser.add_argument('--symbol', type=str, help='Analyze single symbol')
    parser.add_argument('--min-score', type=int, default=60, help='Minimum alert score (default: 60)')
    parser.add_argument('--no-pushover', action='store_true', help='Disable Pushover notifications')
    
    args = parser.parse_args()
    
    # Load Pushover config
    app_token, user_key = None, None
    if not args.no_pushover:
        app_token, user_key = load_pushover_config()
        if app_token and user_key:
            print("✅ Pushover notifications enabled")
        else:
            print("⚠️  Pushover not configured")
    
    scanner = PreBreakoutScanner()
    
    print(f"🔍 Pre-Breakout Scanner - Monthly Timeframe Alert System")
    print(f"Target: Find stocks BEFORE they breakout (like HOOD in early 2023)")
    print(f"Timeframe: Monthly bars for institutional signals")
    print("=" * 70)
    
    if args.symbol:
        # Analyze single symbol
        print(f"Analyzing {args.symbol.upper()}...")
        setup = scanner.analyze_monthly_setup(args.symbol.upper())
        
        if setup:
            scanner.display_monthly_alerts([setup])
            
            # Send Pushover alert for single symbol if score >= 50
            if setup['alert_score'] >= 50 and app_token and user_key:
                title = f"📈 {setup['symbol']} Alert"
                message = f"Score: {setup['alert_score']:.0f}\nAction: {setup['monthly_action']}\nPrice: ${setup['current_price']:.2f}"
                send_pushover_alert(app_token, user_key, title, message)
        else:
            print(f"❌ No monthly setup detected for {args.symbol.upper()}")
        return
    
    # Get universe to scan
    if args.focused:
        universe = get_focused_universe()
        print(f"📊 Running FOCUSED scan ({len(universe)} stocks) - All stocks treated equally")
    else:
        universe = get_comprehensive_universe()
        print(f"📊 Running COMPREHENSIVE scan ({len(universe)} stocks) - All stocks treated equally")
    
    # Scan for monthly setups
    alerts = scanner.scan_universe_for_monthly_setups(universe)
    
    # Filter by minimum score
    filtered_alerts = [a for a in alerts if a['alert_score'] >= args.min_score]
    
    # Send Pushover alerts for scores >= 50
    pushover_alerts = [a for a in alerts if a['alert_score'] >= 50]
    if pushover_alerts and app_token and user_key:
        # Send summary alert
        strong_alerts = [a for a in pushover_alerts if a['alert_score'] >= 80]
        
        if strong_alerts:
            title = f"🚨 {len(strong_alerts)} Strong Alerts Found!"
            symbols = [f"{a['symbol']} ({a['alert_score']:.0f})" for a in strong_alerts[:5]]  # Top 5
            message = f"Top alerts:\n" + "\n".join(symbols)
            if len(strong_alerts) > 5:
                message += f"\n...and {len(strong_alerts)-5} more"
            send_pushover_alert(app_token, user_key, title, message, priority=1)
        elif len(pushover_alerts) >= 3:
            title = f"📈 {len(pushover_alerts)} Alerts Found"
            symbols = [f"{a['symbol']} ({a['alert_score']:.0f})" for a in pushover_alerts[:3]]
            message = "Alerts:\n" + "\n".join(symbols)
            send_pushover_alert(app_token, user_key, title, message)
    
    # Display results
    scanner.display_monthly_alerts(filtered_alerts)
    
    # Summary
    if filtered_alerts:
        print(f"\n💡 SUMMARY:")
        print(f"Found {len(filtered_alerts)} monthly setups with score >= {args.min_score}")
        
        # Group by action
        strong_buy = [a for a in filtered_alerts if a['alert_score'] >= 85]
        buy = [a for a in filtered_alerts if 80 <= a['alert_score'] < 85]
        accumulate = [a for a in filtered_alerts if 70 <= a['alert_score'] < 80]
        watch = [a for a in filtered_alerts if 60 <= a['alert_score'] < 70]
        
        if strong_buy:
            print(f"🚨 STRONG BUY ({len(strong_buy)}): {', '.join([a['symbol'] for a in strong_buy])}")
        if buy:
            print(f"🔥 BUY ({len(buy)}): {', '.join([a['symbol'] for a in buy])}")
        if accumulate:
            print(f"📈 ACCUMULATE ({len(accumulate)}): {', '.join([a['symbol'] for a in accumulate])}")
        if watch:
            print(f"👀 WATCH ({len(watch)}): {', '.join([a['symbol'] for a in watch])}")
        
        print(f"\n📱 Next steps:")
        print(f"1. Add high-scoring stocks to watchlist")
        print(f"2. Monitor monthly for breakout confirmations")
        print(f"3. Start with small positions on 80+ scores")
        
    else:
        print(f"\n💡 No setups found with score >= {args.min_score}")
        print(f"Try lowering --min-score or check back later")

if __name__ == "__main__":
    main()