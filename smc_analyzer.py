"""
SMC Analyzer (Smart Money Concepts)
Implementasi lengkap: Market Structure (BOS/CHoCH), Order Blocks, FVG, 
Liquidity Sweeps, dan Premium/Discount Zones.
Dependencies: pip install yfinance pandas numpy matplotlib
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SMCAnalyzer:
    def __init__(self, symbol: str = "BTC-USD", period: str = "3mo", interval: str = "1h", swing_window: int = 5):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.swing_window = swing_window
        self.df = None
        self.pd_range = None

    def fetch_data(self) -> pd.DataFrame:
        """Ambil data OHLCV dari yfinance"""
        print(f"📡 Mengambil data {self.symbol} ({self.period} | {self.interval})...")
        df = yf.download(self.symbol, period=self.period, interval=self.interval, auto_adjust=True, progress=False)
        # Bersihkan kolom multi-index jika ada
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.dropna(inplace=True)
        self.df = df
        return self.df

    def detect_swings(self):
        """Deteksi Swing High & Swing Low berdasarkan window"""
        df = self.df
        w = self.swing_window
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan

        for i in range(w, len(df) - w):
            if df['high'].iloc[i] > df['high'].iloc[i-w:i].max() and df['high'].iloc[i] > df['high'].iloc[i+1:i+w+1].max():
                df.at[df.index[i], 'swing_high'] = df['high'].iloc[i]
            if df['low'].iloc[i] < df['low'].iloc[i-w:i].min() and df['low'].iloc[i] < df['low'].iloc[i+1:i+w+1].min():
                df.at[df.index[i], 'swing_low'] = df['low'].iloc[i]
        self.df = df

    def detect_fvg(self):
        """Deteksi Fair Value Gaps (Imbalance)"""
        df = self.df
        # Bullish FVG: low[i] > high[i-2]
        df['bullish_fvg'] = (df['low'].shift(0) > df['high'].shift(2))
        # Bearish FVG: high[i] < low[i-2]
        df['bearish_fvg'] = (df['high'].shift(0) < df['low'].shift(2))
        # Simpan batas zona FVG untuk plotting
        df['fvg_top'] = np.where(df['bullish_fvg'], df['high'].shift(2), np.where(df['bearish_fvg'], df['high'], np.nan))
        df['fvg_bottom'] = np.where(df['bullish_fvg'], df['low'], np.where(df['bearish_fvg'], df['low'].shift(2), np.nan))
        self.df = df

    def detect_structure_signals(self):
        """Deteksi BOS & CHoCH berdasarkan urutan Swing High/Low"""
        df = self.df
        df['signal'] = np.nan
        df['trend'] = np.nan

        swings = df[['swing_high', 'swing_low']].dropna()
        if len(swings) < 3:
            print("⚠️ Data swing tidak cukup untuk mendeteksi struktur.")
            return

        trend = 'neutral'
        last_bos_high = None
        last_bos_low = None

        for idx in swings.index:
            val_high = df.at[idx, 'swing_high']
            val_low = df.at[idx, 'swing_low']

            if pd.notna(val_high):
                if trend == 'bear':
                    # Break struktur bearish -> CHoCH Bullish
                    if last_bos_high is None or val_high > last_bos_high:
                        df.at[idx, 'signal'] = 'CHoCH_BULL'
                        trend = 'bull'
                        last_bos_high = val_high
                elif trend in ['neutral', 'bull']:
                    if last_bos_high is not None and val_high > last_bos_high:
                        df.at[idx, 'signal'] = 'BOS_BULL'
                    last_bos_high = val_high
                    trend = 'bull'

            elif pd.notna(val_low):
                if trend == 'bull':
                    # Break struktur bullish -> CHoCH Bearish
                    if last_bos_low is None or val_low < last_bos_low:
                        df.at[idx, 'signal'] = 'CHoCH_BEAR'
                        trend = 'bear'
                        last_bos_low = val_low
                elif trend in ['neutral', 'bear']:
                    if last_bos_low is not None and val_low < last_bos_low:
                        df.at[idx, 'signal'] = 'BOS_BEAR'
                    last_bos_low = val_low
                    trend = 'bear'

        df['trend'] = trend
        self.df = df

    def detect_liquidity_sweeps(self):
        """Deteksi Liquidity Sweep (Fake-out di swing high/low)"""
        df = self.df
        df['sweep_high'] = False
        df['sweep_low'] = False

        # Ambil swing terakhir yang valid sebelum index saat ini
        swing_highs = df['swing_high'].dropna()
        swing_lows = df['swing_low'].dropna()

        for i in range(1, len(df)):
            prev_sh = swing_highs[swing_highs.index < df.index[i]].max() if len(swing_highs[swing_highs.index < df.index[i]]) > 0 else np.nan
            prev_sl = swing_lows[swing_lows.index < df.index[i]].min() if len(swing_lows[swing_lows.index < df.index[i]]) > 0 else np.nan

            # Sweep High: Wick menembus swing high, tapi close di bawahnya
            if pd.notna(prev_sh) and df['high'].iloc[i] > prev_sh and df['close'].iloc[i] < prev_sh:
                df.at[df.index[i], 'sweep_high'] = True
            # Sweep Low: Wick menembus swing low, tapi close di atasnya
            if pd.notna(prev_sl) and df['low'].iloc[i] < prev_sl and df['close'].iloc[i] > prev_sl:
                df.at[df.index[i], 'sweep_low'] = True
        self.df = df

    def calculate_premium_discount(self):
        """Hitung Zona Premium, Equilibrium, dan Discount"""
        df = self.df
        recent_highs = df['swing_high'].dropna()
        recent_lows = df['swing_low'].dropna()
        
        if len(recent_highs) > 0 and len(recent_lows) > 0:
            range_high = recent_highs.max()
            range_low = recent_lows.min()
            eq = (range_high + range_low) / 2
            
            df['equilibrium'] = eq
            df['in_discount'] = df['close'] < eq
            df['in_premium'] = df['close'] > eq
            self.pd_range = (range_low, range_high, eq)
        self.df = df

    def run_all(self):
        """Jalankan seluruh pipeline analisis"""
        self.fetch_data()
        self.detect_swings()
        self.detect_structure_signals()
        self.detect_fvg()
        self.detect_liquidity_sweeps()
        self.calculate_premium_discount()
        print("✅ Analisis SMC selesai.")
        return self.df

    def visualize(self):
        """Visualisasi hasil analisis SMC"""
        df = self.df
        if df is None: return

        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot Close Price
        ax.plot(df.index, df['close'], label='Close Price', color='#1f77b4', linewidth=1.5)

        # Plot FVG Zones
        fvg_data = df[df['bullish_fvg'] | df['bearish_fvg']]
        for _, row in fvg_data.iterrows():
            if row['bullish_fvg']:
                ax.axhspan(row['high'].shift(2), row['low'], color='green', alpha=0.2)
            else:
                ax.axhspan(row['low'].shift(2), row['high'], color='red', alpha=0.2)

        # Plot Premium / Discount Zones
        if self.pd_range:
            low, high, eq = self.pd_range
            ax.axhline(eq, color='gray', linestyle='--', alpha=0.8, label='Equilibrium')
            ax.axhspan(low, eq, alpha=0.1, color='blue', label='Discount Zone (Buy)')
            ax.axhspan(eq, high, alpha=0.1, color='orange', label='Premium Zone (Sell)')

        # Plot BOS & CHoCH
        signals = df[df['signal'].notna()]
        for _, row in signals.iterrows():
            color = 'green' if 'BULL' in str(row['signal']) else 'red'
            ax.plot(row.name, df['close'].loc[row.name], '^' if 'BULL' in str(row['signal']) else 'v', 
                    color=color, markersize=10, label=row['signal'])

        # Plot Liquidity Sweeps
        sweeps_high = df[df['sweep_high']]
        sweeps_low = df[df['sweep_low']]
        ax.plot(sweeps_high.index, sweeps_high['high'], 'x', color='darkorange', markersize=8, label='Liquidity Sweep High')
        ax.plot(sweeps_low.index, sweeps_low['low'], 'x', color='purple', markersize=8, label='Liquidity Sweep Low')

        ax.set_title(f'📊 Smart Money Concepts Analysis: {self.symbol}', fontsize=16, pad=10)
        ax.set_ylabel('Price', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.show()

# ==========================================
# 🚀 EKSEKUSI UTAMA
# ==========================================
if __name__ == "__main__":
    # Inisialisasi Analyzer
    smc = SMCAnalyzer(
        symbol="BTC-USD",  # Ganti dengan symbol lain: "AAPL", "EURUSD=X", dll.
        period="3mo",
        interval="1h",     # 1m, 5m, 15m, 1h, 1d
        swing_window=5     # Sensitivitas swing (3-10 optimal)
    )
    
    # Jalankan pipeline
    df_result = smc.run_all()
    
    # Tampilkan ringkasan sinyal terbaru
    print("\n📈 Ringkasan Sinyal Terakhir:")
    summary_cols = ['open', 'high', 'low', 'close', 'swing_high', 'swing_low', 'signal', 'trend']
    print(df_result[summary_cols].dropna(subset=['signal']).tail(5).to_string())
    
    # Tampilkan chart
    smc.visualize()
