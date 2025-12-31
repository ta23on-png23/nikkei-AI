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

# --- ページ設定 ---
st.set_page_config(page_title="日本株AI統合分析ツール", layout="wide")
st.title('🇯🇵 日本株AI統合分析ツール (根拠コメント付き)')

# --- 確率計算関数 ---
def calculate_probability(current_price, predicted_price, lower_bound, upper_bound):
    c, p, l, u = to_float(current_price), to_float(predicted_price), to_float(lower_bound), to_float(upper_bound)
    sigma = (u - l) / 2.56
    if sigma == 0: return 50.0
    z_score = (p - c) / sigma
    return norm.cdf(z_score) * 100

# --- ★新機能：AIの根拠生成関数 ---
def get_ai_reasons(forecast, current_date, target_date, current_price, predicted_price):
    reasons = []
    
    # データを取得
    # current_row = forecast.loc[forecast['ds'] == pd.to_datetime(current_date)].iloc[0] # 直近は予測データに含まれないことがあるため省略
    target_row = forecast.iloc[(forecast['ds'] - target_date).abs().argsort()[:1]].iloc[0]
    
    # 1. トレンド分析 (trend成分)
    # 予測価格と現在価格の差を見るのが一番確実
    price_diff_pct = ((predicted_price - current_price) / current_price) * 100
    
    if price_diff_pct > 5.0:
        reasons.append("📈 **強い上昇トレンド**: 長期的な成長軌道に乗っており、AIは力強い上昇を予測しています。")
    elif price_diff_pct > 0:
        reasons.append("↗️ **緩やかな上昇**: 急騰ではありませんが、底堅い上昇トレンドが継続すると判断しました。")
    elif price_diff_pct < -5.0:
        reasons.append("📉 **下落警戒**: 長期トレンドが下向きであり、AIは慎重な見方をしています。")
    else:
        reasons.append("➡️ **横ばい**: 明確なトレンドが出ておらず、現在の価格帯での推移を予測しています。")

    # 2. 季節性分析 (yearly成分)
    # その時期が、年間を通して「高い時期」か「低い時期」か
    if 'yearly' in target_row:
        yearly_effect = target_row['yearly']
        if yearly_effect > 0:
            reasons.append("🌸 **季節性の追い風**: 例年、この時期は株価が上がりやすい傾向（アノマリー）があります。")
        elif yearly_effect < 0:
            reasons.append("🍂 **季節性の向かい風**: 例年、この時期は調整局面に入りやすい傾向があります。")

    # 3. 曜日要因 (weekly成分)
    # 短期（1ヶ月以内）の場合のみ表示
    days_diff = (target_date - current_date).days
    if days_diff <= 30 and 'weekly' in target_row:
        weekly_effect = target_row['weekly']
        week_day_name = target_date.strftime('%A') # 曜日名
        if weekly_effect > 0:
            reasons.append(f"📅 **曜日要因**: この銘柄は統計的に「{week_day_name}」に強い傾向があります。")
            
    return reasons

# ==========================================
#  PART 1: 有望株スクリーニング
# ==========================================
st.header("1️⃣ 有望株AIスクリーニング (上昇確率85%以上)")
st.markdown("複数の銘柄を一括チェックします。")

default_tickers = "7203, 9984, 8306, 7974, 6920"
user_tickers = st.text_area("銘柄コードリスト (カンマ区切り)", default_tickers, height=70)

if st.button('🚀 リスト作成開始 (5社推奨)'):
    ticker_list = [t.strip() for t in user_tickers.split(',') if t.strip()]
    results = []
    my_bar = st.progress(0, text="AIが計算中...")
    
    for i, code in enumerate(ticker_list):
        my_bar.progress((i + 1) / len(ticker_list), text=f"計算中: {code}")
        try:
            t_symbol = f"{code}.T"
            df_hist = yf.download(t_symbol, period="3y", interval="1d", progress=False)
            if len(df_hist) > 100:
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
                
                m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
                m.fit(df_p)
                fut = m.make_future_dataframe(periods=366, freq='D')
                fcst = m.predict(fut)
                
                probs = {}
                tgt_days = {"3ヶ月": 90, "6ヶ月": 180, "12ヶ月": 365}
                is_hot = False
                
                for lbl, d in tgt_days.items():
                    tgt_d = last_d + timedelta(days=d)
                    diff = (fcst['ds'] - tgt_d).abs()
                    c_idx = diff.argsort()[:1]
                    cl = fcst.iloc[c_idx].iloc[0]
                    pv = calculate_probability(curr, to_float(cl['yhat']), to_float(cl['yhat_lower']), to_float(cl['yhat_upper']))
                    probs[lbl] = pv
                    if pv >= 85.0: is_hot = True
                
                results.append({
                    "コード": code,
                    "現在値": f"{curr:,.0f}",
                    "3ヶ月確率": probs["3ヶ月"],
                    "6ヶ月確率": probs["6ヶ月"],
                    "12ヶ月確率": probs["12ヶ月"],
                    "判定": "🔥 激熱" if is_hot else "-"
                })
        except: continue
    
    my_bar.empty()
    if results:
        res_df = pd.DataFrame(results)
        def highlight(val):
            return f'background-color: #ffcccc; color: black' if isinstance(val, float) and val >= 85.0 else ''
        st.dataframe(res_df.style.applymap(highlight, subset=["3ヶ月確率", "6ヶ月確率", "12ヶ月確率"]).format("{:.1f}%"), use_container_width=True)

st.markdown("---")

# ==========================================
#  PART 2: 個別詳細分析
# ==========================================
st.header("2️⃣ 個別銘柄 詳細分析 & AI根拠")
st.markdown("AIがなぜその予測を出したのか、**根拠（トレンド・季節性）**も表示します。")

col_input, col_btn = st.columns([3, 1])
with col_input:
    detail_code = st.text_input("分析する銘柄コード (例: 7203)", "7203")
with col_btn:
    st.write("") 
    st.write("")
    start_detail = st.button('📊 詳細分析スタート')

if start_detail:
    ticker = f"{detail_code}.T"
    try:
        with st.spinner(f'{detail_code} を詳細分析中...'):
            stk_data = yf.download(ticker, period="5y", interval="1d", progress=False)
            usd_data = yf.download("USDJPY=X", period="5y", interval="1d", progress=False)

        if stk_data.empty:
            st.error("データが見つかりません。")
            st.stop()

        def clean_df(raw_df):
            df = raw_df.reset_index()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            cols = {c.lower(): c for c in df.columns}
            d_c = next((c for k, c in cols.items() if 'date' in k), df.columns[0])
            c_c = next((c for k, c in cols.items() if 'close' in k), df.columns[1])
            o_c = next((c for k, c in cols.items() if 'open' in k), c_c)
            h_c = next((c for k, c in cols.items() if 'high' in k), c_c)
            l_c = next((c for k, c in cols.items() if 'low' in k), c_c)
            out = pd.DataFrame()
            out['ds'] = pd.to_datetime(df[d_c]).dt.tz_localize(None)
            out['Open'] = df[o_c]
            out['High'] = df[h_c]
            out['Low'] = df[l_c]
            out['Close'] = df[c_c]
            return out

        df_s = clean_df(stk_data)
        df_u = clean_df(usd_data)

        # 銘柄名
        try:
            info = yf.Ticker(ticker)
            name = info.info.get('longName', f"コード: {detail_code}")
        except: name = f"コード: {detail_code}"

        curr_price = to_float(df_s['Close'].iloc[-1])
        last_dt = df_s['ds'].iloc[-1]

        st.subheader(f"🏢 {name}")
        st.metric("現在終値", f"{curr_price:,.0f} 円", f"基準日: {last_dt.strftime('%Y/%m/%d')}")

        # A. 急変動チェック
        st.subheader("⚡ 過去の急変動 (5%以上) と要因")
        df_s['Change'] = df_s['Close'].pct_change() * 100
        df_u['Change'] = df_u['Close'].pct_change() * 100
        df_m = pd.merge(df_s, df_u[['ds', 'Change']], on='ds', how='inner', suffixes=('', '_USD'))
        big_moves = df_m[df_m['Change'].abs() >= 5.0].copy().sort_values('ds', ascending=False)

        if not big_moves.empty:
            m_res = []
            for idx, row in big_moves.iterrows():
                d_str = row['ds'].strftime('%Y-%m-%d')
                move = "急騰" if row['Change'] > 0 else "急落"
                url = f"https://www.google.com/search?q={name} {d_str} 株価 {move} 理由"
                u_chg = row['Change_USD']
                corr = "🔄 連動?" if (row['Change']*u_chg > 0 and abs(u_chg)>0.5) else "⚡ 独自"
                m_res.append({"日時": d_str, "変動率": f"{row['Change']:+.2f}%", "ドル円": f"{u_chg:+.2f}%", "タイプ": corr, "詳細": url})
            st.dataframe(pd.DataFrame(m_res), column_config={"詳細": st.column_config.LinkColumn("ニュース検索", display_text="🔍 理由")}, hide_index=True)
        else:
            st.info("※ 直近5年間で、日足5%以上の急変動はありませんでした。")

        # B. AI予測と根拠
        with st.spinner('AIが未来を予測中...'):
            df_prophet = pd.DataFrame({'ds': df_s['ds'], 'y': df_s['Close']})
            m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=366, freq='D')
            forecast = m.predict(future)

        st.subheader('🎯 未来の上昇・下落確率とAIの根拠')
        fut_fcst = forecast[forecast['ds'] > last_dt].copy()
        targets = {"1ヶ月後": 30, "3ヶ月後": 90, "6ヶ月後": 180, "12ヶ月後": 365}
        
        for lbl, days in targets.items():
            tgt_d = last_dt + timedelta(days=days)
            diff = (fut_fcst['ds'] - tgt_d).abs()
            c_idx = diff.argsort()[:1]
            if len(c_idx) > 0:
                row = fut_fcst.iloc[c_idx].iloc[0]
                pred = to_float(row['yhat'])
                pup = calculate_probability(curr_price, pred, to_float(row['yhat_lower']), to_float(row['yhat_upper']))
                
                # AI根拠の取得
                reasons = get_ai_reasons(forecast, last_dt, tgt_d, curr_price, pred)
                
                trend = "➡️ レンジ"
                if pup >= 60: trend = "↗️ 上昇優勢"
                elif 100-pup >= 60: trend = "↘️ 下落優勢"

                # 表示用コンテナ
                with st.container():
                    st.markdown(f"### 🕒 **{lbl}** の予測 ({row['ds'].strftime('%Y/%m/%d')})")
                    c1, c2, c3 = st.columns([1, 1, 2])
                    c1.metric("予測株価", f"{pred:,.0f} 円")
                    c2.metric("上昇確率", f"{pup:.1f} %", trend)
                    with c3:
                        st.markdown("**AIの判断根拠:**")
                        for r in reasons:
                            st.markdown(f"- {r}")
                    st.divider()

        # C. チャート
        st.subheader('📊 長期予測チャート')
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df_s['ds'], open=df_s['Open'], high=df_s['High'], low=df_s['Low'], close=df_s['Close'], name='実測値'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='AI予測', line=dict(color='yellow', width=2)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)', hoverinfo='skip', showlegend=False, name='予測範囲'))
        fig.update_layout(title=f"{name} 日足チャート & AI予測", template="plotly_dark", height=600, xaxis_rangeslider_visible=True)
        fig.update_xaxes(range=[last_dt - timedelta(days=365), last_dt + timedelta(days=365)])
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
