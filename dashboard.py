"""
dashboard.py - Integrated Fundamental & SMC Analysis Dashboard
Dashboard interaktif dengan Streamlit + Plotly untuk analisis saham lengkap
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import analyzer modules
from fundamental_analyzer import FundamentalAnalyzer, analyze_stock, print_report_summary
from smc_analyzer import SMCAnalyzer

# ============================================================================
# CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="📊 Stock Analysis Pro Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stAlert {
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - INPUT PARAMETERS
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Stock Input
    stock_symbol = st.text_input(
        "📝 Stock Symbol",
        value="AAPL",
        help="Enter stock ticker (e.g., AAPL, BBCA.JK, GOOGL)"
    )
    
    currency = st.selectbox(
        "💰 Currency",
        options=["USD", "IDR", "EUR", "GBP"],
        index=0
    )
    
    st.divider()
    
    # SMC Parameters
    st.subheader("📐 SMC Settings")
    smc_period = st.selectbox(
        "Time Period",
        options=["1mo", "3mo", "6mo", "1y", "2y"],
        index=1
    )
    
    smc_interval = st.selectbox(
        "Interval",
        options=["1d", "1h", "15m"],
        index=0
    )
    
    swing_window = st.slider(
        "Swing Sensitivity",
        min_value=3,
        max_value=10,
        value=5
    )
    
    st.divider()
    
    # Risk-free rate for DCF
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5
    )
    
    # Monte Carlo iterations
    mc_iterations = st.slider(
        "Monte Carlo Iterations",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )
    
    st.divider()
    
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    else:
        st.session_state.run_analysis = False
    
    # Info box
    with st.expander("ℹ️ About This Dashboard"):
        st.markdown("""
        **Dashboard ini menggabungkan:**
        
        1. **Fundamental Analysis**
           - 57+ Rasio Keuangan
           - Piotroski F-Score
           - Altman Z-Score
           - DCF Valuation dengan Monte Carlo
        
        2. **Smart Money Concepts (SMC)**
           - Market Structure (BOS/CHoCH)
           - Fair Value Gaps (FVG)
           - Liquidity Sweeps
           - Premium/Discount Zones
        
        **Data Source:** Yahoo Finance
        """)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown('<p class="main-header">📊 Stock Analysis Pro Dashboard</p>', unsafe_allow_html=True)
st.markdown("Integrated Fundamental & Technical Analysis with Smart Money Concepts")

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'smc_result' not in st.session_state:
    st.session_state.smc_result = None

# Run analysis when button is clicked or on first load with default symbol
if st.session_state.get('run_analysis', False) or st.session_state.analysis_result is None:
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # ========== FUNDAMENTAL ANALYSIS ==========
        status_text.text("🔄 Running Fundamental Analysis...")
        
        analyzer = FundamentalAnalyzer(stock_symbol, currency=currency, risk_free_rate=risk_free_rate/100)
        fundamental_report = analyzer.generate_full_report()
        
        progress_bar.progress(25)
        status_text.text("✅ Fundamental Analysis Complete")
        
        # ========== SMC ANALYSIS ==========
        status_text.text("🔄 Running SMC Analysis...")
        
        smc = SMCAnalyzer(
            symbol=stock_symbol,
            period=smc_period,
            interval=smc_interval,
            swing_window=swing_window
        )
        smc_df = smc.run_all()
        
        progress_bar.progress(50)
        status_text.text("✅ SMC Analysis Complete")
        
        # Store results
        st.session_state.analysis_result = fundamental_report
        st.session_state.smc_result = {
            'df': smc_df,
            'pd_range': smc.pd_range,
            'symbol': stock_symbol
        }
        
        progress_bar.progress(100)
        status_text.text("✅ All Analysis Complete!")
        
        st.success(f"Analysis completed for **{stock_symbol}** at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.info("💡 Please check the stock symbol and try again.")
        st.stop()

# Load results from session state
fundamental_report = st.session_state.analysis_result
smc_data = st.session_state.smc_result

if fundamental_report is None:
    st.warning("⚠️ Please click 'Run Analysis' to start.")
    st.stop()

# ============================================================================
# TABS FOR DIFFERENT SECTIONS
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "💹 Fundamental Ratios",
    "🎯 Scoring & Valuation",
    "📈 SMC Analysis",
    "📊 Charts"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    st.header("📋 Company Overview & Key Metrics")
    
    # Company Info Row
    meta = fundamental_report['metadata']
    market_data = fundamental_report['market_data']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Company",
            value=meta.get('company_name', 'N/A')[:30],
            delta=f"{meta.get('sector', 'Unknown')} Sector"
        )
    
    with col2:
        current_price = market_data.get('current_price', 0)
        st.metric(
            label="Current Price",
            value=f"{currency} {current_price:,.2f}" if current_price else "N/A",
            delta=f"52W High: {market_data.get('52week_range', (0,0))[1]:,.2f}" if current_price else None
        )
    
    with col3:
        market_cap = market_data.get('market_cap', 0)
        if market_cap:
            if market_cap > 1e12:
                cap_str = f"{market_cap/1e12:.2f}T"
            elif market_cap > 1e9:
                cap_str = f"{market_cap/1e9:.2f}B"
            else:
                cap_str = f"{market_cap/1e6:.2f}M"
            st.metric(label="Market Cap", value=cap_str)
        else:
            st.metric(label="Market Cap", value="N/A")
    
    with col4:
        beta = market_data.get('beta', 0)
        st.metric(
            label="Beta",
            value=f"{beta:.2f}" if beta else "N/A",
            delta="High Volatility" if beta and beta > 1.5 else ("Low Volatility" if beta and beta < 0.8 else "Average")
        )
    
    st.divider()
    
    # Overall Assessment
    assessment = fundamental_report['overall_assessment']
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("🎯 Overall Rating")
        
        # Display rating with color
        rating_color = {
            'STRONG BUY': '🟢',
            'BUY': '🟡',
            'HOLD': '🟠',
            'REDUCE': '🔴',
            'SELL/AVOID': '⚫'
        }
        
        rating = assessment.get('rating', 'N/A')
        score = assessment.get('composite_score', 0)
        
        # Create gauge chart for score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Composite Score /100", 'font': {'size': 16}},
            delta={'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "royalblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#feebe2'},
                    {'range': [20, 40], 'color': '#fcc5c0'},
                    {'range': [40, 60], 'color': '#fa9fb5'},
                    {'range': [60, 80], 'color': '#ff7fbf'},
                    {'range': [80, 100], 'color': '#ae017e'}
                ],
            }
        ))
        
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"### {rating_color.get(rating, '')} {rating}")
        st.caption(f"Score: {score}/100")
    
    with col2:
        st.subheader("✅ Key Strengths")
        strengths = assessment.get('key_strengths', [])
        if strengths:
            for strength in strengths:
                st.markdown(f"✓ {strength}")
        else:
            st.write("No specific strengths identified")
    
    with col3:
        st.subheader("⚠️ Key Risks")
        risks = assessment.get('key_risks', [])
        if risks:
            for risk in risks:
                st.markdown(f"⚠ {risk}")
        else:
            st.write("No specific risks identified")
    
    st.divider()
    
    # Quick Metrics Grid
    st.subheader("📊 Quick Fundamental Metrics")
    
    valuation = fundamental_report['valuation_ratios']
    profitability = fundamental_report['profitability_ratios']
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    metrics = [
        (col1, "P/E Ratio", valuation.get('pe_ratio_ttm'), ""),
        (col2, "P/BV", valuation.get('pbv_ratio'), ""),
        (col3, "ROE", profitability.get('roe'), "%"),
        (col4, "Net Margin", profitability.get('net_margin'), "%"),
        (col5, "Div Yield", valuation.get('dividend_yield'), "%"),
        (col6, "Beta", market_data.get('beta'), ""),
    ]
    
    for col, label, value, suffix in metrics:
        with col:
            if value:
                if suffix == "%":
                    display_val = f"{value*100:.1f}{suffix}" if value < 1 else f"{value:.1f}{suffix}"
                else:
                    display_val = f"{value:.2f}" if isinstance(value, float) else str(value)
                st.metric(label=label, value=display_val)
            else:
                st.metric(label=label, value="N/A")

# ============================================================================
# TAB 2: FUNDAMENTAL RATIOS
# ============================================================================

with tab2:
    st.header("💹 Detailed Fundamental Ratios")
    
    # Create sub-tabs for different ratio categories
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "Valuation", "Profitability", "Liquidity & Solvency", "Efficiency & Growth"
    ])
    
    with sub_tab1:
        st.subheader("🏷️ Valuation Ratios")
        
        valuation = fundamental_report['valuation_ratios']
        
        # Create DataFrame for display
        val_df = pd.DataFrame({
            'Metric': list(valuation.keys()),
            'Value': [f"{v:.4f}" if v else "N/A" for v in valuation.values()]
        })
        
        st.dataframe(val_df, use_container_width=True, hide_index=True)
        
        # Benchmark comparison
        benchmark = fundamental_report.get('sector_benchmarking', {})
        if benchmark.get('comparison'):
            st.subheader("📊 vs Sector Benchmark")
            
            comp_data = []
            for metric, data in benchmark['comparison'].items():
                comp_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': f"{data.get('value', 'N/A'):.2f}" if data.get('value') else "N/A",
                    'Benchmark': f"{data.get('benchmark', 'N/A')}",
                    'Status': data.get('status', 'N/A')
                })
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    with sub_tab2:
        st.subheader("💰 Profitability Ratios")
        
        profitability = fundamental_report['profitability_ratios']
        
        prof_df = pd.DataFrame({
            'Metric': list(profitability.keys()),
            'Value': [f"{v*100:.2f}%" if v else "N/A" for v in profitability.values()]
        })
        
        st.dataframe(prof_df, use_container_width=True, hide_index=True)
        
        # Visualization
        if any(v for v in profitability.values() if v):
            fig = go.Figure()
            
            metrics_to_plot = ['roe', 'roa', 'roic', 'gross_margin', 'operating_margin', 'net_margin']
            values = []
            labels = []
            
            for m in metrics_to_plot:
                if m in profitability and profitability[m]:
                    values.append(profitability[m] * 100)
                    labels.append(m.replace('_', ' ').title())
            
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color='rgb(55, 83, 109)',
                text=[f"{v:.1f}%" for v in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Profitability Metrics (%)",
                xaxis_title="Metric",
                yaxis_title="Percentage (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with sub_tab3:
        st.subheader("💧 Liquidity & Solvency Ratios")
        
        liquidity = fundamental_report['liquidity_solvency_ratios']
        
        liq_df = pd.DataFrame({
            'Metric': list(liquidity.keys()),
            'Value': [f"{v:.4f}" if v else "N/A" for v in liquidity.values()]
        })
        
        st.dataframe(liq_df, use_container_width=True, hide_index=True)
    
    with sub_tab4:
        st.subheader("⚡ Efficiency & Growth Ratios")
        
        efficiency = fundamental_report['efficiency_growth_ratios']
        
        eff_df = pd.DataFrame({
            'Metric': list(efficiency.keys()),
            'Value': [f"{v*100:.2f}%" if v and abs(v) < 1 else f"{v:.4f}" if v else "N/A" 
                     for v in efficiency.values()]
        })
        
        st.dataframe(eff_df, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 3: SCORING & VALUATION
# ============================================================================

with tab3:
    st.header("🎯 Scoring Models & Valuation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Piotroski F-Score")
        
        fscore_data = fundamental_report['piotroski_fscore']
        fscore = fscore_data.get('fscore', 0)
        interpretation = fscore_data.get('interpretation', {})
        
        # F-Score Gauge
        fig_fscore = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fscore,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "F-Score (0-9)", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [None, 9], 'tickwidth': 2},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'steps': [
                    {'range': [0, 3], 'color': '#ffcccc'},
                    {'range': [3, 6], 'color': '#ffffcc'},
                    {'range': [6, 9], 'color': '#ccffcc'}
                ],
            }
        ))
        
        fig_fscore.update_layout(height=300)
        st.plotly_chart(fig_fscore, use_container_width=True)
        
        st.markdown(f"**Category:** {interpretation.get('category', 'N/A')}")
        st.markdown(f"**Signal:** {interpretation.get('signal', 'N/A')}")
        st.markdown(f"**Risk Level:** {interpretation.get('risk', 'N/A')}")
        
        # Details
        with st.expander("📋 F-Score Components"):
            details = fscore_data.get('details', {})
            for component, value in details.items():
                icon = "✅" if value == 1 else "❌"
                st.write(f"{icon} {component.replace('_', ' ').title()}: {value}")
    
    with col2:
        st.subheader("⚠️ Altman Z-Score")
        
        zscore_data = fundamental_report['altman_zscore']
        
        if 'z_score' in zscore_data:
            zscore = zscore_data.get('z_score', 0)
            zone = zscore_data.get('zone', 'Unknown')
            
            # Color based on zone
            zone_colors = {
                'Safe Zone': 'green',
                'Grey Zone': 'orange',
                'Distress Zone': 'red'
            }
            
            fig_zscore = go.Figure(go.Indicator(
                mode="gauge+number",
                value=zscore,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Z-Score", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [None, 5], 'tickwidth': 2},
                    'bar': {'color': zone_colors.get(zone, 'blue')},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'steps': [
                        {'range': [0, 1.81], 'color': '#ffcccc'},
                        {'range': [1.81, 2.99], 'color': '#ffffcc'},
                        {'range': [2.99, 5], 'color': '#ccffcc'}
                    ],
                    'thresholds': [
                        {'value': 1.81, 'line': {'color': "red", 'width': 4}, 'text': "Distress"},
                        {'value': 2.99, 'line': {'color': "green", 'width': 4}, 'text': "Safe"}
                    ]
                }
            ))
            
            fig_zscore.update_layout(height=300)
            st.plotly_chart(fig_zscore, use_container_width=True)
            
            st.markdown(f"**Zone:** {zone}")
            st.markdown(f"**Bankruptcy Risk:** {zscore_data.get('bankruptcy_risk', 'N/A')}")
            st.markdown(f"**Signal:** {zscore_data.get('signal', 'N/A')}")
            
            # Components
            with st.expander("📋 Z-Score Components"):
                components = zscore_data.get('components', {})
                for comp, value in components.items():
                    st.write(f"**{comp.upper()}:** {value}")
        else:
            st.warning(zscore_data.get('error', 'Z-Score calculation failed'))
    
    st.divider()
    
    # DCF Valuation
    st.subheader("💵 DCF Valuation with Monte Carlo Simulation")
    
    dcf_data = fundamental_report['dcf_valuation']
    
    if 'intrinsic_value_mean' in dcf_data:
        col1, col2, col3 = st.columns(3)
        
        intrinsic_mean = dcf_data.get('intrinsic_value_mean', 0)
        intrinsic_median = dcf_data.get('intrinsic_value_median', 0)
        current_price = dcf_data.get('current_price', 0)
        upside = dcf_data.get('upside_downside_pct', 0)
        ci = dcf_data.get('confidence_interval_95', (0, 0))
        
        with col1:
            st.metric(
                "Intrinsic Value (Mean)",
                f"{currency} {intrinsic_mean:,.2f}",
                delta=f"{upside:+.1f}% vs Current"
            )
        
        with col2:
            st.metric(
                "Intrinsic Value (Median)",
                f"{currency} {intrinsic_median:,.2f}"
            )
        
        with col3:
            st.metric(
                "95% CI",
                f"{currency} {ci[0]:,.0f} - {ci[1]:,.0f}"
            )
        
        # DCF Distribution Chart
        fig_dcf = go.Figure()
        
        # Add distribution (simulated normal distribution)
        std = dcf_data.get('intrinsic_value_std', intrinsic_median * 0.2)
        x_vals = np.linspace(intrinsic_mean - 3*std, intrinsic_mean + 3*std, 100)
        y_vals = 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*((x_vals-intrinsic_mean)/std)**2)
        
        fig_dcf.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            fill='tozeroy',
            name='Valuation Distribution',
            line=dict(color='royalblue')
        ))
        
        # Add current price line
        fig_dcf.add_vline(
            x=current_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current: {currency} {current_price:,.2f}"
        )
        
        # Add intrinsic value line
        fig_dcf.add_vline(
            x=intrinsic_median,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Intrinsic: {currency} {intrinsic_median:,.2f}"
        )
        
        fig_dcf.update_layout(
            title="DCF Valuation Distribution (Monte Carlo)",
            xaxis_title="Intrinsic Value",
            yaxis_title="Probability Density",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_dcf, use_container_width=True)
        
        # Decision
        decision = dcf_data.get('valuation_decision', {})
        st.info(f"**Recommendation:** {decision.get('action', 'N/A')} - {decision.get('signal', 'N/A')}")
        
        # Parameters used
        with st.expander("📋 DCF Parameters"):
            params = dcf_data.get('parameters_used', {})
            for param, value in params.items():
                if isinstance(value, float):
                    st.write(f"**{param.replace('_', ' ').title()}:** {value:.4f}")
                else:
                    st.write(f"**{param.replace('_', ' ').title()}:** {value}")
    else:
        st.warning(dcf_data.get('error', 'DCF calculation failed'))

# ============================================================================
# TAB 4: SMC ANALYSIS
# ============================================================================

with tab4:
    st.header("📈 Smart Money Concepts Analysis")
    
    if smc_data and smc_data['df'] is not None:
        df_smc = smc_data['df']
        
        # Summary Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        signals = df_smc[df_smc['signal'].notna()]
        fvg_count = len(df_smc[df_smc['bullish_fvg'] | df_smc['bearish_fvg']])
        sweeps_high = len(df_smc[df_smc['sweep_high']])
        sweeps_low = len(df_smc[df_smc['sweep_low']])
        
        with col1:
            st.metric("Structure Signals", len(signals))
        with col2:
            st.metric("FVG Patterns", fvg_count)
        with col3:
            st.metric("High Sweeps", sweeps_high)
        with col4:
            st.metric("Low Sweeps", sweeps_low)
        
        st.divider()
        
        # Current Trend
        current_trend = df_smc['trend'].iloc[-1] if 'trend' in df_smc.columns else 'Unknown'
        trend_emoji = {'bull': '🐂', 'bear': '🐻', 'neutral': '➖'}.get(current_trend, '❓')
        st.markdown(f"### Current Trend: {trend_emoji} {current_trend.upper()}")
        
        # Premium/Discount Zone
        if smc_data['pd_range']:
            low, high, eq = smc_data['pd_range']
            current_close = df_smc['Close'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Range Low", f"{low:.2f}")
            with col2:
                st.metric("Equilibrium", f"{eq:.2f}")
            with col3:
                st.metric("Range High", f"{high:.2f}")
            
            # Determine position
            if current_close < eq:
                position = "DISCOUNT ZONE (Potential Buy)"
                position_color = "🟢"
            elif current_close > eq:
                position = "PREMIUM ZONE (Potential Sell)"
                position_color = "🔴"
            else:
                position = "AT EQUILIBRIUM"
                position_color = "🟡"
            
            st.info(f"{position_color} Current Price ({current_close:.2f}): {position}")
        
        st.divider()
        
        # Recent Signals Table
        st.subheader("🔔 Recent Structure Signals")
        
        if len(signals) > 0:
            signal_cols = ['Date', 'Close', 'swing_high', 'swing_low', 'signal', 'trend']
            available_cols = [c for c in signal_cols if c in df_smc.columns]
            
            signal_display = signals[available_cols].tail(10).copy()
            
            # Format for display
            for col in ['Close', 'swing_high', 'swing_low']:
                if col in signal_display.columns:
                    signal_display[col] = signal_display[col].round(2)
            
            st.dataframe(signal_display, use_container_width=True, hide_index=True)
        else:
            st.warning("No structure signals detected in the selected timeframe.")
        
        st.divider()
        
        # FVG Details
        st.subheader("📊 Fair Value Gaps")
        
        bullish_fvg = df_smc[df_smc['bullish_fvg']].tail(5)
        bearish_fvg = df_smc[df_smc['bearish_fvg']].tail(5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🟢 Recent Bullish FVG**")
            if len(bullish_fvg) > 0:
                for idx, row in bullish_fvg.iterrows():
                    st.write(f"• {row.get('Date', idx)}: {row.get('Low', 0):.2f} - {row.get('High', 0):.2f}")
            else:
                st.write("No recent bullish FVG")
        
        with col2:
            st.markdown("**🔴 Recent Bearish FVG**")
            if len(bearish_fvg) > 0:
                for idx, row in bearish_fvg.iterrows():
                    st.write(f"• {row.get('Date', idx)}: {row.get('Low', 0):.2f} - {row.get('High', 0):.2f}")
            else:
                st.write("No recent bearish FVG")
    else:
        st.warning("SMC data not available. Please run the analysis again.")

# ============================================================================
# TAB 5: CHARTS
# ============================================================================

with tab5:
    st.header("📊 Interactive Charts")
    
    if smc_data and smc_data['df'] is not None:
        df_smc = smc_data['df']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price with SMC Signals', 'Volume', 'RSI')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df_smc['Date'],
                open=df_smc['Open'],
                high=df_smc['High'],
                low=df_smc['Low'],
                close=df_smc['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Swing High/Low markers
        swing_highs = df_smc[df_smc['swing_high'].notna()]
        swing_lows = df_smc[df_smc['swing_low'].notna()]
        
        if len(swing_highs) > 0:
            fig.add_trace(
                go.Scatter(
                    x=swing_highs['Date'],
                    y=swing_highs['swing_high'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='red'),
                    name='Swing High',
                    hoverinfo='text',
                    text=[f"Swing High: {v:.2f}" for v in swing_highs['swing_high']]
                ),
                row=1, col=1
            )
        
        if len(swing_lows) > 0:
            fig.add_trace(
                go.Scatter(
                    x=swing_lows['Date'],
                    y=swing_lows['swing_low'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='green'),
                    name='Swing Low',
                    hoverinfo='text',
                    text=[f"Swing Low: {v:.2f}" for v in swing_lows['swing_low']]
                ),
                row=1, col=1
            )
        
        # Add BOS/CHoCH signals
        signals = df_smc[df_smc['signal'].notna()]
        if len(signals) > 0:
            bull_signals = signals[signals['signal'].str.contains('BULL')]
            bear_signals = signals[signals['signal'].str.contains('BEAR')]
            
            if len(bull_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=bull_signals['Date'],
                        y=bull_signals['Low'],
                        mode='markers+text',
                        marker=dict(symbol='arrow-up', size=15, color='green'),
                        name='Bullish Signal',
                        text=bull_signals['signal'],
                        textposition='bottom center'
                    ),
                    row=1, col=1
                )
            
            if len(bear_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=bear_signals['Date'],
                        y=bear_signals['High'],
                        mode='markers+text',
                        marker=dict(symbol='arrow-down', size=15, color='red'),
                        name='Bearish Signal',
                        text=bear_signals['signal'],
                        textposition='top center'
                    ),
                    row=1, col=1
                )
        
        # Add Premium/Discount zones
        if smc_data['pd_range']:
            low, high, eq = smc_data['pd_range']
            
            # Discount zone
            fig.add_hrect(
                y0=low, y1=eq,
                fillcolor="rgba(0, 255, 0, 0.1)",
                line_width=0,
                annotation_text="Discount Zone",
                annotation_position="right",
                row=1, col=1
            )
            
            # Premium zone
            fig.add_hrect(
                y0=eq, y1=high,
                fillcolor="rgba(255, 0, 0, 0.1)",
                line_width=0,
                annotation_text="Premium Zone",
                annotation_position="right",
                row=1, col=1
            )
            
            # Equilibrium line
            fig.add_hline(
                y=eq,
                line_dash="dash",
                line_color="gray",
                annotation_text="Equilibrium",
                row=1, col=1
            )
        
        # Volume chart
        colors = ['green' if df_smc['Close'].iloc[i] >= df_smc['Open'].iloc[i] else 'red' 
                  for i in range(len(df_smc))]
        
        fig.add_trace(
            go.Bar(
                x=df_smc['Date'],
                y=df_smc['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Calculate RSI
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        rsi = calculate_rsi(df_smc['Close'])
        
        fig.add_trace(
            go.Scatter(
                x=df_smc['Date'],
                y=rsi,
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{smc_data['symbol']} - Comprehensive Technical Analysis",
            height=900,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional charts
        st.subheader("📈 Additional Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig_hist = px.histogram(
                df_smc,
                x='Close',
                nbins=30,
                title='Price Distribution',
                labels={'Close': 'Price'},
                color_discrete_sequence=['royalblue']
            )
            fig_hist.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Volume vs Price scatter
            fig_scatter = px.scatter(
                df_smc,
                x='Close',
                y='Volume',
                title='Volume vs Price',
                labels={'Close': 'Price', 'Volume': 'Volume'},
                color='Volume',
                color_continuous_scale='Viridis'
            )
            fig_scatter.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Chart data not available. Please run the analysis again.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p><strong>📊 Stock Analysis Pro Dashboard</strong></p>
    <p>Built with Streamlit + Plotly | Data Source: Yahoo Finance</p>
    <p><em>Disclaimer: This dashboard is for educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)
