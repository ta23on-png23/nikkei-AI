import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from scipy.stats import norm
import plotly.graph_objs as go
from datetime import timedelta

# ==========================================
#  設定：パスワード
# ==========================================
# ★ここに設定したいパスワードを入れてください（今は demo です）
DEMO_PASSWORD = "demo" 

# --- ページ設定 (必ず一番最初に書く) ---
st.set_page_config(page_title="東P株AIツール", layout="wide")

# --- UI非表示 & 黒背景デザイン (CSS) ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            div[data-testid="stToolbar"] {visibility: hidden;}
            .stDeployButton {display:none;}
            
            /* アプリ全体の背景黒・文字白 */
            .stApp {
                background-color: #000000;
                color: #ffffff;
            }
            /* 全てのテキストを白く・太く */
            h1, h2, h3, h4, h5, h6, p, div, span, label, li {
                color: #ffffff !important;
                font-family: sans-serif;
            }
            /* ラジオボタン */
            div[data-testid="stRadio"] label p {
                font-weight: bold !important;
                font-size: 1.1rem !important;
                color: #ffffff !important;
            }
            /* 入力ボックス */
            .stTextInput > div > div > input {
                color: #ffffff !important;
                background-color: #333333;
                font-weight: bold;
            }
            /* 余白調整 */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 5rem;
                padding-left: 0.5rem;
                padding-right: 0.5rem;
            }
            /* テーブルヘッダー非表示 */
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- パスワード認証機能 ---
def check_password():
    """パスワードが正しいかチェックする関数"""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    # パスワード入力画面
    st.markdown("### 🔒 デモ版アクセス制限")
    password = st.text_input("パスワードを入力してください", type="password")
    
    if password:
        if password == DEMO_PASSWORD:
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("パスワードが違います")
    return False

# 認証チェック（通らないとここで止まる）
if not check_password():
    st.stop()

# ==========================================
#  メインアプリ処理
# ==========================================

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

# --- タイトル & デモ版表記 ---
st.markdown("### **東P株AIツール**")
st.markdown("""
<div style="margin-top: -15px; margin-bottom: 10px;">
    <span style="font-size: 0.9rem;">デモ版</span><br>
    <span style="font-size: 0.7rem; opacity: 0.8;">（※数値は過去データーから予測されたもので、結果を保証するものではありません）</span>
</div>
""", unsafe_allow_html=True)

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

# --- AI要因判定関数 ---
def get_ai_reasons_short(forecast, target_date, current_price, predicted_price):
    tags = []
    target_row = forecast.iloc[(forecast['ds'] - target_date).abs().argsort()[:1]].iloc[0]
    
    diff_pct = ((predicted_price - current_price) / current_price) * 100
    if diff_pct > 5.0: tags.append("上昇トレンド")
    elif diff_pct > 0: tags.append("緩やかな上昇")
    elif diff_pct < -5.0: tags.append("下落/調整局面")
    else: tags.append("レンジ/横ばい")

    if 'yearly' in target_row:
        y_eff = target_row['yearly']
        if y_eff > 0: tags.append("季節性(良)")
        elif y_eff < 0: tags.append("季節性(悪)")
    
    return "・".join(tags)

# --- メイン処理 ---
if target_code:
    ticker = f"{target_code}.T"
    
    # データのダウンロードとエラーハンドリング
    try:
        with st.spinner('AIが計算中...'):
            df_hist = yf.download(ticker, period=period_str, interval="1d", progress=False)
        
        # 【重要】データが空っぽだった場合の処理を追加
        if df_hist.empty:
            st.error(f"データが見つかりません。コード「{target_code}」を確認してください。")
            st.stop()

        # データ数が少なすぎる場合
        if len(df_hist) < 30:
            st.error("分析に必要なデータ数が足りません（上場直後などの可能性があります）。")
            st.stop()

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
                reasons = get_ai_reasons_short(fcst, tgt_d, curr, pred)
                
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

        # --- 5. 長期予測チャート (ボタン追加版) ---
        st.markdown("#### **長期予測チャート**")

        zoom_period = st.radio(
            "表示期間",
            ["1M", "3M", "6M", "1Y", "3Y", "ALL"],
            index=3,
            horizontal=True,
            label_visibility="collapsed"
        )

        days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "3Y": 365*3, "ALL": 365*5}
        
        if zoom_period == "ALL":
            x_start = df_hist[date_c].min()
            x_end = fcst['ds'].max()
        else:
            days = days_map[zoom_period]
            x_start = last_d - timedelta(days=days)
            x_end = last_d + timedelta(days=days) 
            
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_hist[date_c], open=df_hist['Open'], high=df_hist['High'], low=df_hist['Low'], close=df_hist['Close'], name='実測'))
        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], mode='lines', name='AI', line=dict(color='yellow', width=2)))
        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)', hoverinfo='skip', showlegend=False))

        fig.update_layout(
            template="plotly_dark",
            height=450, 
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(
                rangeslider=dict(visible=False), 
                range=[x_start, x_end],
                type="date",
                fixedrange=True
            ),
            yaxis=dict(
                fixedrange=True
            ),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'staticPlot': False, 'scrollZoom': False})

    except Exception as e:
        st.error(f"システムエラーが発生しました: {e}")
