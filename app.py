import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from scipy.stats import norm
from datetime import timedelta

# --- 安全な数値変換関数 ---
def to_float(x):
    try:
        if isinstance(x, float): return x
        if isinstance(x, (pd.Series, pd.DataFrame)):
            if x.empty: return 0.0
            return float(x.to_numpy()[0])
        if hasattr(x, 'item'): return float(x.item())
        if isinstance(x, list): return float(x[0])
        return float(x)
    except: return 0.0

# --- ページ設定 ---
st.set_page_config(page_title="日本株AI有望株リスト", layout="wide")
st.title('🇯🇵 日本株AI有望株リスト (数値特化版)')

# --- 確率計算関数 ---
def calculate_probability(current_price, predicted_price, lower_bound, upper_bound):
    c, p, l, u = to_float(current_price), to_float(predicted_price), to_float(lower_bound), to_float(upper_bound)
    sigma = (u - l) / 2.56
    if sigma == 0: return 50.0
    z_score = (p - c) / sigma
    return norm.cdf(z_score) * 100

# ==========================================
#  PART 1: 有望株スクリーニング (リスト作成)
# ==========================================
st.header("🔍 有望株AIリスト作成 (上昇確率85%以上)")
st.markdown("監視したい銘柄コードを入力してください。**3ヶ月・6ヶ月・12ヶ月後**の上昇確率を計算し、リスト化します。")

# デフォルト銘柄 (5社)
default_tickers = "7203, 9984, 8306, 7974, 6920"
user_tickers = st.text_area("銘柄コードリスト (カンマ区切り)", default_tickers, height=70)

if st.button('🚀 リスト作成開始 (5社推奨)'):
    ticker_list = [t.strip() for t in user_tickers.split(',') if t.strip()]
    
    results = []
    progress_text = "AIが各銘柄を計算中..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, code in enumerate(ticker_list):
        my_bar.progress((i + 1) / len(ticker_list), text=f"計算中: {code} ({i+1}/{len(ticker_list)})")
        
        try:
            t_symbol = f"{code}.T"
            # 過去データ取得 (グラフ描画しないのでデータ処理は最小限でOK)
            df_hist = yf.download(t_symbol, period="3y", interval="1d", progress=False)
            
            if len(df_hist) > 100:
                df_hist = df_hist.reset_index()
                if isinstance(df_hist.columns, pd.MultiIndex):
                    df_hist.columns = df_hist.columns.get_level_values(0)
                
                cols = {c.lower(): c for c in df_hist.columns}
                date_c = next((c for k, c in cols.items() if 'date' in k), df_hist.columns[0])
                close_c = next((c for k, c in cols.items() if 'close' in k), df_hist.columns[1])
                
                df_prophet = pd.DataFrame()
                df_prophet['ds'] = pd.to_datetime(df_hist[date_c]).dt.tz_localize(None)
                df_prophet['y'] = df_hist[close_c]
                
                current_price = to_float(df_prophet['y'].iloc[-1])
                last_date = df_prophet['ds'].iloc[-1]
                
                # AI学習
                m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                m.fit(df_prophet)
                
                # 1年先まで予測
                future = m.make_future_dataframe(periods=366, freq='D')
                forecast = m.predict(future)
                
                # 確率計算
                probs = {}
                target_days = {"3ヶ月": 90, "6ヶ月": 180, "12ヶ月": 365}
                is_promising = False
                
                for label, days in target_days.items():
                    target_date = last_date + timedelta(days=days)
                    
                    # --- 【修正箇所】長い行を分割してエラー回避 ---
                    # 日付の差分を計算
                    time_diff = (forecast['ds'] - target_date).abs()
                    # 一番近い日のインデックスを取得
                    closest_idx = time_diff.argsort()[:1]
                    # その行のデータを取得
                    closest = forecast.iloc[closest_idx].iloc[0]
                    # ----------------------------------------
                    
                    p_val = calculate_probability(
                        current_price, 
                        to_float(closest['yhat']), 
                        to_float(closest['yhat_lower']), 
                        to_float(closest['yhat_upper'])
                    )
                    probs[label] = p_val
                    
                    if p_val >= 85.0:
                        is_promising = True
                
                # 銘柄名取得 (簡易)
                try:
                    ticker_info = yf.Ticker(t_symbol)
                    comp_name = ticker_info.info.get('longName', code)
                except:
                    comp_name = code

                results.append({
                    "コード": code,
                    "銘柄名": comp_name,
                    "現在値": f"{current_price:,.0f}",
                    "3ヶ月確率": probs["3ヶ月"],
                    "6ヶ月確率": probs["6ヶ月"],
                    "12ヶ月確率": probs["12ヶ月"],
                    "判定": "🔥 激熱" if is_promising else "-"
                })

        except Exception:
            continue

    my_bar.empty()

    if results:
        res_df = pd.DataFrame(results)
        
        # 85%以上を赤く塗るデザイン設定
        def highlight_high_prob(val):
            color = '#ffcccc' if isinstance(val, float) and val >= 85.0 else ''
            return f'background-color: {color}; color: black'

        st.subheader("📋 分析結果リスト")
        st.dataframe(
            res_df.style.applymap(highlight_high_prob, subset=["3ヶ月確率", "6ヶ月確率", "12ヶ月確率"])
                  .format({"3ヶ月確率": "{:.1f}%", "6ヶ月確率": "{:.1f}%", "12ヶ月確率": "{:.1f}%"}),
            use_container_width=True,
            height=500
        )
    else:
        st.warning("データが取得できませんでした。")

st.markdown("---")
st.caption("※ チャート表示機能を削除し、リスト表示に特化した軽量版です。")
