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

st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem; /* 下部に少し余裕を持たせる */
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stTextInput > div > div > input {
        color: white;
        background-color: #333333;
    }
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
    """, unsafe_allow_html=True)

# --- タイトル ---
st.markdown("**東P株AIツール**")

# --- 期間選択（3年、5年ボタン） ---
period_label = st.radio(
    "期間選択",
    ("3年", "5年"),
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)
# "3年" -> 3 に変換
period_select = int(period_label.replace("年", ""))
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
                short_name = full_name[:4] + "..." if len(full_name) > 4 else full_name
            except:
                full_name = target_code
                short_name = target_code

            # --- 1. AIスクリーニングデータ ---
            st.markdown("**AIスクリーニングデーター**")
            
            probs = {}
            tgt_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
            
            for lbl, d in tgt_map.items():
                tgt_d = last_d + timedelta(days=d)
                diff = (fcst['ds'] - tgt_d).abs()
                c_idx = diff.argsort()[:1]
                cl = fcst.iloc[c_idx].iloc[0]
                pv = calculate_probability(curr, to_float(cl['yhat']), to_float(cl['yhat_lower']), to_float(cl['yhat_upper']))
                probs[lbl] = pv

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

            # --- 2. 詳細情報 ---
            st.markdown(f"**{full_name}**")
            st.write(f"現在値: {curr:,.0f} 円")

            # --- 3. 過去の変動要因 ---
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

            # --- 4. 未来の予測 ---
            fut_fcst = fcst[fcst['ds'] > last_d].copy()
            for lbl, days in tgt_map.items():
                tgt_d = last_d + timedelta(days=days)
                diff = (fut_fcst['ds'] - tgt_d).abs()
                c_idx = diff.argsort()[:1]
                if len(c_idx) > 0:
                    row = fut_fcst.iloc[c_idx].iloc[0]
                    pred = to_float(row['yhat'])
                    pup = calculate_probability(curr, pred, to_float(row['yhat_lower']), to_float(row['yhat_upper']))
                    st.write(f"**{lbl}後の予測**: {pred:,.0f}円  {pup:.1f}%")

            # --- 5. 長期予測チャート (ボタン追加版) ---
            st.markdown("**長期予測チャート**")
            fig = go.Figure()
            # 実測
            fig.add_trace(go.Candlestick(x=df_hist[date_c], open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name='実測'))
            # AI予測
            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], mode='lines', name='AI', line=dict(color='yellow', width=2)))
            # 予測帯
            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)', hoverinfo='skip', showlegend=False))

            # --- チャート用ボタン設定 ---
            # 期間に応じてボタンを調整 (3年データなら5Yボタンは不要だが、あっても害はないので汎用的に配置)
            # スマホで見やすいように、チャートの頭にボタンを置く
            fig.update_layout(
                template="plotly_dark",
                height=450, # ボタンが入る分少し高さを確保
                margin=dict(l=0, r=0, t=50, b=0), # 上部ボタン用にスペース確保
                xaxis=dict(
                    rangeslider=dict(visible=False), # 下のスライダーは非表示
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(count=3, label="3Y", step="year", stepmode="backward"),
                            dict(count=5, label="5Y", step="year", stepmode="backward"),
                            dict(step="all", label="All")
                        ]),
                        font=dict(color="black"), # ボタン文字色
                        bgcolor="#eeeeee",        # ボタン背景色
                        activecolor="#ff9900"     # 選択中の色
                    ),
                    type="date"
                ),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("データ取得エラー")
