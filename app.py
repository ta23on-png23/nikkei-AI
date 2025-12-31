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
st.set_page_config(page_title="日本株AI予測（長期）", layout="wide")
st.title('🇯🇵 日本株プライム AI長期予測＆急変動分析')

# --- サイドバー：銘柄入力 ---
st.sidebar.header("銘柄設定")
stock_code = st.sidebar.text_input("銘柄コード (例: 7203)", "7203")
ticker = f"{stock_code}.T"

if st.sidebar.button('🔄 分析開始'):
    st.rerun()

st.sidebar.markdown("""
**期間設定: 日足 (Daily)**
長期予測のため、日単位のデータを使用します。

**表示の見方**
- **上昇確率**: 現在価格より上がる確率
- **急変動**: 過去に5%以上動いた日
""")

# --- 確率計算関数 ---
def calculate_probability(current_price, predicted_price, lower_bound, upper_bound):
    c, p, l, u = to_float(current_price), to_float(predicted_price), to_float(lower_bound), to_float(upper_bound)
    sigma = (u - l) / 2.56
    if sigma == 0: return 50.0
    z_score = (p - c) / sigma
    return norm.cdf(z_score) * 100

# --- メイン処理 ---
try:
    # 1. データ取得 (期間を長く、間隔を日足に変更)
    with st.spinner(f'{stock_code} の過去データ（5年分）を取得・分析中...'):
        # 長期予測のため過去5年分の日足を取得
        stock_data = yf.download(ticker, period="5y", interval="1d", progress=False)
        usdjpy_data = yf.download("USDJPY=X", period="5y", interval="1d", progress=False)

    if stock_data.empty:
        st.error(f"データが見つかりません。コード {stock_code} が正しいか確認してください。")
        st.stop()

    # --- データ整形 ---
    def clean_df(raw_df):
        df = raw_df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        cols = {c.lower(): c for c in df.columns}
        date_c = next((c for k, c in cols.items() if 'date' in k), df.columns[0])
        close_c = next((c for k, c in cols.items() if 'close' in k), df.columns[1])
        open_c = next((c for k, c in cols.items() if 'open' in k), close_c)
        high_c = next((c for k, c in cols.items() if 'high' in k), close_c)
        low_c = next((c for k, c in cols.items() if 'low' in k), close_c)

        df_out = pd.DataFrame()
        df_out['ds'] = pd.to_datetime(df[date_c]).dt.tz_localize(None)
        df_out['Open'] = df[open_c]
        df_out['High'] = df[high_c]
        df_out['Low'] = df[low_c]
        df_out['Close'] = df[close_c]
        return df_out

    df_stock = clean_df(stock_data)
    df_usdjpy = clean_df(usdjpy_data)

    # 銘柄名取得
    try:
        ticker_info = yf.Ticker(ticker)
        stock_name = ticker_info.info.get('longName', f"コード: {stock_code}")
    except:
        stock_name = f"コード: {stock_code}"

    latest_close = to_float(df_stock['Close'].iloc[-1])
    latest_time = df_stock['ds'].iloc[-1]

    # --- 2. 画面トップ表示 ---
    st.subheader(f"🏢 {stock_name} (日足分析)")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label="現在終値", value=f"{latest_close:,.1f} 円")
    with col2:
        st.info(f"データ基準日: {latest_time.strftime('%Y/%m/%d')}")

    # --- 3. 急変動チェック (日足で5%以上) ---
    st.subheader("⚡ 過去の急変動 (5%以上) と要因")
    
    df_stock['Change'] = df_stock['Close'].pct_change() * 100
    df_merged = pd.merge(df_stock, df_usdjpy[['ds', 'Change']], on='ds', how='inner', suffixes=('', '_USD'))
    
    threshold = 5.0 # 5%以上の変動を検知
    big_moves = df_merged[df_merged['Change'].abs() >= threshold].copy().sort_values('ds', ascending=False)

    if not big_moves.empty:
        move_results = []
        for index, row in big_moves.iterrows():
            date_str = row['ds'].strftime('%Y-%m-%d')
            change_val = row['Change']
            usd_change = row['Change_USD']
            
            # 要因診断
            if (change_val > 0 and usd_change > 0.5) or (change_val < 0 and usd_change < -0.5):
                correlation = "🔄 ドル円連動の可能性"
            else:
                correlation = "⚡ 個別材料の可能性大"

            move_type = "急騰" if change_val > 0 else "急落"
            search_query = f"{stock_name} {date_str} 株価 {move_type} 理由"
            search_url = f"https://www.google.com/search?q={search_query}"

            move_results.append({
                "日時": row['ds'].strftime('%Y/%m/%d'),
                "変動率": f"{change_val:+.2f}%",
                "ドル円": f"{usd_change:+.2f}%",
                "AI簡易診断": correlation,
                "詳細調査": search_url
            })
        
        st.dataframe(
            pd.DataFrame(move_results),
            column_config={
                "詳細調査": st.column_config.LinkColumn("ニュース検索", display_text="🔍 Googleで検索")
            },
            hide_index=True
        )
    else:
        st.write(f"※ 直近5年間で、日足ベースで {threshold}% 以上動いた日はありませんでした。")

    # --- 4. AI予測 (1年先まで) ---
    with st.spinner('AIが1年先まで予測計算中...'):
        df_prophet = pd.DataFrame({'ds': df_stock['ds'], 'y': df_stock['Close']})
        # 日足用設定: daily_seasonality=False(日内変動なし), yearly_seasonality=True(年間の季節性あり)
        m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(df_prophet)
        
        # 1年分(365日)の未来枠を作成
        future = m.make_future_dataframe(periods=365, freq='D')
        forecast = m.predict(future)

    # --- ターゲット確率表示 (3日, 1週, 1月, 1年) ---
    st.subheader('🎯 未来の上昇・下落確率')
    
    # 未来データ抽出
    future_forecast = forecast[forecast['ds'] > latest_time].copy()
    
    # ターゲット日数設定
    target_days = {
        "3日後": 3,      # ここを1→3に変更しました
        "1週間後": 7,
        "1か月後": 30,
        "1年後": 365
    }
    
    prob_results = []
    
    for label, days in target_days.items():
        # 目標日付を計算
        target_date = latest_time + timedelta(days=days)
        
        # 目標日付に最も近い予測データを検索
        closest_row = future_forecast.iloc[(future_forecast['ds'] - target_date).abs().argsort()[:1]]
        
        if not closest_row.empty:
            row = closest_row.iloc[0]
            pred_val = to_float(row['yhat'])
            prob_up = calculate_probability(latest_close, pred_val, to_float(row['yhat_lower']), to_float(row['yhat_upper']))
            
            trend = "➡️ レンジ"
            if prob_up >= 60: trend = "↗️ 上昇優勢"
            elif 100-prob_up >= 60: trend = "↘️ 下落優勢"
            
            prob_results.append({
                "期間": label,
                "予測時期": row['ds'].strftime('%Y/%m/%d'),
                "現在株価": f"{latest_close:,.0f}",
                "予測株価": f"{pred_val:,.0f}",
                "上昇確率": f"{prob_up:.1f} %",
                "下落確率": f"{100-prob_up:.1f} %",
                "判定": trend
            })

    st.table(pd.DataFrame(prob_results).set_index("期間"))

    # --- 5. チャート表示 ---
    st.subheader('📊 予測推移チャート (日足)')
    fig = go.Figure()
    
    # 過去データ(ローソク足)
    fig.add_trace(go.Candlestick(
        x=df_stock['ds'],
        open=df_stock['Open'], high=df_stock['High'],
        low=df_stock['Low'], close=df_stock['Close'],
        name='実測値', increasing_line_color='#00CC96', decreasing_line_color='#EF553B'
    ))
    
    # AI予測
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='AI予測ライン', line=dict(color='yellow', width=2)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)', hoverinfo='skip', showlegend=False, name='予測範囲'))

    fig.update_layout(
        title=f"{stock_name} 長期予測チャート",
        yaxis_title="株価 (円)",
        template="plotly_dark", height=600, xaxis_rangeslider_visible=True
    )
    # 直近1年+未来1年くらいにズーム
    start_zoom = latest_time - timedelta(days=365)
    end_zoom = latest_time + timedelta(days=365)
    fig.update_xaxes(range=[start_zoom, end_zoom])
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"エラーが発生しました: {e}")