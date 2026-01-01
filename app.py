import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from scipy.stats import norm
import plotly.graph_objs as go
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

# --- ページ設定 & 黒背景CSS適用 ---
st.set_page_config(page_title="東P株AIツール", layout="wide")

# CSSで黒背景、余白削除、文字サイズ調整
st.markdown("""
    <style>
    /* 全体の背景を黒に */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    /* タイトル周りの余白を詰める */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* 入力フォームの背景調整 */
    .stTextInput > div > div > input {
        color: white;
        background-color: #333333;
    }
    /* テーブルのヘッダー調整 */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
    """, unsafe_allow_html=True)

# --- タイトル ---
st.markdown("**東P株AIツール**")

# --- 期間選択（メイン画面配置） ---
period_select = st.radio(
    "期間選択",
    (3, 5),
    index=0,
    horizontal=True,
    label_visibility="collapsed" # ラベルを隠してシンプルに
)
period_str = f"{period_select}y"
st.write(f"※ 過去{period_select}年データで分析")

# --- 入力エリア ---
target_code = st.text_input("銘柄コード", "7203")

# --- 確率計算関数 ---
def calculate_probability(current_price, predicted_price, lower_bound, upper_bound):
    c, p, l, u = to_float(current_price), to_float(predicted_price), to_float(lower_bound), to_float(upper_bound)
    sigma = (u - l) / 2.56
    if sigma == 0: return 50.0
    z_score = (p - c) / sigma
    return norm.cdf(z_score) * 100

# --- メイン処理 ---
if target_code:
    ticker = f"{target_code}.T"
    
    # データ取得
    try:
        with st.spinner('Loading...'):
            df_hist = yf.download(ticker, period=period_str, interval="1d", progress=False)
            
        if len(df_hist) > 50:
            # --- データ整形 ---
            df_hist = df_hist.reset_index()
            if isinstance(df_hist.columns, pd.MultiIndex):
                df_hist.columns = df_hist.columns.get_level_values(0)
            
            cols = {c.lower(): c for c in df_hist.columns}
            date_c = next((c for k, c in cols.items() if 'date' in k), df_hist.columns[0])
            close_c = next((c for k, c in cols.items() if 'close' in k), df_hist.columns[1])
            
            df_p = pd.DataFrame()
            df_p['ds'] = pd.to_datetime(df_hist[date_c]).dt.tz_localize(None)
            df_p['y'] = df_hist[close_c]
            
            curr = to_float(df_p['y'].iloc[-1])
            last_d = df_p['ds'].iloc[-1]
            
            # --- AI予測 ---
            m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(df_p)
            fut = m.make_future_dataframe(periods=366, freq='D')
            fcst = m.predict(fut)

            # --- 企業名取得と短縮 ---
            try:
                info = yf.Ticker(ticker)
                full_name = info.info.get('longName', target_code)
                # 4文字程度に短縮
                short_name = full_name[:4] + "..." if len(full_name) > 4 else full_name
            except:
                full_name = target_code
                short_name = target_code

            # --- 1. AIスクリーニングデータ (表) ---
            st.markdown("**AIスクリーニングデーター**")
            
            probs = {}
            # 表示名を変更 (1M, 3M, 6M, 1Y)
            tgt_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
            
            for lbl, d in tgt_map.items():
                tgt_d = last_d + timedelta(days=d)
                diff = (fcst['ds'] - tgt_d).abs()
                c_idx = diff.argsort()[:1]
                cl = fcst.iloc[c_idx].iloc[0]
                pv = calculate_probability(curr, to_float(cl['yhat']), to_float(cl['yhat_lower']), to_float(cl['yhat_upper']))
                probs[lbl] = pv

            # 1行だけのデータフレーム作成
            screen_data = [{
                "コード": target_code,
                "企業名": short_name,
                "現在値": f"{curr:,.0f}",
                "1M": f"{probs['1M']:.1f}%",
                "3M": f"{probs['3M']:.1f}%",
                "6M": f"{probs['6M']:.1f}%",
                "1Y": f"{probs['1Y']:.1f}%",
            }]
            st.dataframe(pd.DataFrame(screen_data), hide_index=True, use_container_width=True)

            # --- 2. 詳細情報 (テキスト) ---
            st.markdown(f"**{full_name}**")
            st.write(f"現在値: {curr:,.0f} 円")

            # --- 3. 過去の変動要因 (5%以上) ---
            st.markdown("**過去の変動要因**")
            
            df_hist['Change'] = df_hist[close_c].pct_change() * 100
            big_moves = df_hist[df_hist['Change'].abs() >= 5.0].copy().sort_values(date_c, ascending=False)
            
            if not big_moves.empty:
                m_res = []
                for idx, row in big_moves.iterrows():
                    d_str = row[date_c].strftime('%Y-%m-%d')
                    move = "急騰" if row['Change'] > 0 else "急落"
                    url = f"https://www.google.com/search?q={full_name} {d_str} 株価 {move} 理由"
                    m_res.append({
                        "日時": d_str,
                        "変動率": f"{row['Change']:+.1f}%",
                        "検索": url
                    })
                
                st.dataframe(
                    pd.DataFrame(m_res),
                    column_config={"検索": st.column_config.LinkColumn("検索", display_text="検索")},
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.write("※ 5%以上の変動なし")

            # --- 4. 未来の予測 (テキスト羅列) ---
            # タイトルなしでシンプルに表示
            fut_fcst = fcst[fcst['ds'] > last_d].copy()
            
            for lbl, days in tgt_map.items():
                tgt_d = last_d + timedelta(days=days)
                diff = (fut_fcst['ds'] - tgt_d).abs()
                c_idx = diff.argsort()[:1]
                if len(c_idx) > 0:
                    row = fut_fcst.iloc[c_idx].iloc[0]
                    pred = to_float(row['yhat'])
                    pup = calculate_probability(curr, pred, to_float(row['yhat_lower']), to_float(row['yhat_upper']))
                    
                    # 1行で表示 (例: 1M後の予測: 3000円 91.8%)
                    st.write(f"**{lbl}後の予測**: {pred:,.0f}円  {pup:.1f}%")

            # --- 5. 長期予測チャート ---
            st.markdown("**長期予測チャート**")
            fig = go.Figure()
            # 実測値
            fig.add_trace(go.Candlestick(x=df_hist[date_c], open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name='実測'))
            # AI予測
            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], mode='lines', name='AI', line=dict(color='yellow', width=2)))
            # 帯
            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)', hoverinfo='skip', showlegend=False))

            fig.update_layout(
                template="plotly_dark",
                height=400, # スマホ用に高さを抑える
                margin=dict(l=0, r=0, t=30, b=0), # 余白削除
                xaxis_rangeslider_visible=False, # スライダー削除してスッキリさせる
                showlegend=False
            )
            # ズーム調整
            fig.update_xaxes(range=[last_d - timedelta(days=365), last_d + timedelta(days=365)])
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("データ取得エラー")
