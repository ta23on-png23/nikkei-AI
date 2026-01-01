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

# --- ページ設定 & デザイン調整 (CSS) ---
st.set_page_config(page_title="東P株AIツール", layout="wide")

st.markdown("""
    <style>
    /* 1. アプリ全体の背景黒・文字白 */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* 2. 全てのテキストを白く・太く */
    h1, h2, h3, h4, h5, h6, p, div, span, label, li {
        color: #ffffff !important;
        font-family: sans-serif;
    }
    
    /* 3. ラジオボタン */
    div[data-testid="stRadio"] label p {
        font-weight: bold !important;
        font-size: 1.1rem !important;
        color: #ffffff !important;
    }

    /* 4. 入力ボックス */
    .stTextInput > div > div > input {
        color: #ffffff !important;
        background-color: #333333;
        font-weight: bold;
    }
    
    /* 5. 余白調整 */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    
    /* テーブルヘッダー非表示 */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
    """, unsafe_allow_html=True)

# --- タイトル ---
st.markdown("### **東P株AIツール**")

# --- 期間選択 ---
period_label = st.radio(
    "期間選択",
    ("3年", "5年"),
    index=0,
    horizontal=True,
    label_visibility="collapsed"
)
period_select = int(period_label.replace("年", ""))
period_str = f"{period_select}y"
st.write(f"**※ 過去{period_select}年データで分析**")

# --- 入力エリア ---
target_code = st.text_input("銘柄コード", "7203")

# --- 確率計算関数 ---
def calculate_probability(current_price, predicted_price, lower_bound, upper_bound):
    c, p, l, u = to_float(current_price), to_float(predicted_price), to_float(lower_bound), to_float(upper_bound)
    sigma = (u - l) / 2.56
    if sigma == 0: return 50.0
    z_score = (p - c) / sigma
    return norm.cdf(z_score) * 100

# --- ★AI要因判定関数 (簡潔版) ---
def get_ai_reasons_short(forecast, target_date, current_price, predicted_price):
    tags = []
    # 最も近い予測行を取得
    target_row = forecast.iloc[(forecast['ds'] - target_date).abs().argsort()[:1]].iloc[0]
    
    # 1. トレンド判定
    diff_pct = ((predicted_price - current_price) / current_price) * 100
    if diff_pct > 5.0: tags.append("上昇トレンド")
    elif diff_pct > 0: tags.append("緩やかな上昇")
    elif diff_pct < -5.0: tags.append("下落/調整局面")
    else: tags.append("レンジ/横ばい")

    # 2. 季節性判定
    if 'yearly' in target_row:
        y_eff = target_row['yearly']
        if y_eff > 0: tags.append("季節性(良)")
        elif y_eff < 0: tags.append("季節性(悪)")
    
    return "・".join(tags)

# --- メイン処理 ---
if target_code:
    ticker = f"{target_code}.T"
    
    try:
        with st.spinner('Calculating...'):
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

            # --- 企業名取得 ---
            try:
                info = yf.Ticker(ticker)
                full_name = info.info.get('longName', target_code)
                short_name = full_name[:4] + "..." if len(full_name) > 4 else full_name
            except:
                full_name, short_name = target_code, target_code

            # --- 1. AIスクリーニングデータ ---
            st.markdown("#### **AIスクリーニングデーター**")
            
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
            st.markdown(f"#### **{full_name}**")
            st.write(f"**現在値: {curr:,.0f} 円**")

            # --- 3. 過去の変動要因 ---
            st.markdown("#### **過去の変動要因**")
            df_hist['Change'] = df_hist[close_c].pct_change() * 100
            big_moves = df_hist[df_hist['Change'].abs() >= 5.0].copy().sort_values(date_c, ascending=False)
            
            if not big_moves.empty:
                m_res = []
                for idx, row in big_moves.iterrows():
                    d_str = row[date_c].strftime('%Y-%m-%d')
                    move = "急騰" if row['Change'] > 0 else "急落"
                    url = f"https://www.google.com/search?q={full_name} {d_str} 株価 {move} 理由"
                    m_res.append({"日時": d_str, "変動率": f"{row['Change']:+.1f}%", "検索": url})
                st.dataframe(pd.DataFrame(m_res), column_config={"検索": st.column_config.LinkColumn("検索", display_text="検索")}, hide_index=True, use_container_width=True)
            else:
                st.write("※ 5%以上の変動なし")

            # --- 4. 未来の予測 (要因付き) ---
            st.markdown("#### **未来の予測**")
            fut_fcst = fcst[fcst['ds'] > last_d].copy()
            
            for lbl, days in tgt_map.items():
                tgt_d = last_d + timedelta(days=days)
                diff = (fut_fcst['ds'] - tgt_d).abs()
                c_idx = diff.argsort()[:1]
                if len(c_idx) > 0:
                    row = fut_fcst.iloc[c_idx].iloc[0]
                    pred = to_float(row['yhat'])
                    pup = calculate_probability(curr, pred, to_float(row['yhat_lower']), to_float(row['yhat_upper']))
                    
                    # 要因を取得
                    reasons = get_ai_reasons_short(fcst, tgt_d, curr, pred)
                    
                    # HTMLを使って綺麗にレイアウト (予測値の下に要因)
                    st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <div style="font-size: 1.1rem; font-weight: bold;">
                            {lbl}後の予測: {pred:,.0f}円  {pup:.1f}%
                        </div>
                        <div style="font-size: 0.95rem; color: #dddddd; margin-left: 15px;">
                            └ 要因: {reasons}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # --- 5. 長期予測チャート ---
            st.markdown("#### **長期予測チャート**")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df_hist[date_c], open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name='実測'))
            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], mode='lines', name='AI', line=dict(color='yellow', width=2)))
            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)', hoverinfo='skip', showlegend=False))

            fig.update_layout(
                template="plotly_dark",
                height=450, 
                margin=dict(l=0, r=0, t=50, b=0),
                xaxis=dict(
                    rangeslider=dict(visible=False), 
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
                        font=dict(color="black", size=11), 
                        bgcolor="#eeeeee",        
                        activecolor="#ff9900"     
                    ),
                    type="date"
                ),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"データ取得エラー: {e}")
