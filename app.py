import streamlit as st
import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime
import google.generativeai as genai

# ---------------------------------------------------------
# 【設定エリア】
# ---------------------------------------------------------
st.set_page_config(page_title="底値シグナル分析AI", layout="wide")

# ★Gemini APIの設定
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-pro")
        gemini_available = True
    else:
        gemini_available = False
except:
    gemini_available = False

# ★企業名・業種名のマッピング
NAME_MAP = {
    "1617.T": "食品", "1618.T": "エネ資源", "1619.T": "建設・資材",
    "1620.T": "素材・化学", "1621.T": "医薬品", "1622.T": "自動車・輸送",
    "1623.T": "鉄鋼・非鉄", "1624.T": "機械", "1625.T": "電機・精密",
    "1626.T": "IT・通信", "1627.T": "電力・ガス", "1628.T": "運輸・物流",
    "1629.T": "商社・卸売", "1630.T": "小売", "1631.T": "銀行",
    "1632.T": "金融(除銀行)", "1633.T": "不動産",
    "1326.T": "SPDRゴールド", "1407.T": "ウエストHD", "1419.T": "タマホーム",
    "1489.T": "NF日経高配当50", "1605.T": "INPEX", "1678.T": "NFインド株",
    "2267.T": "ヤクルト", "2516.T": "東証グロース250", "2801.T": "キッコーマン",
    "2897.T": "日清食品HD", "3038.T": "神戸物産", "3099.T": "三越伊勢丹",
    "3382.T": "セブン&アイ", "3397.T": "トリドール", "4045.T": "東亞合成",
    "4543.T": "テルモ", "6758.T": "ソニーG", "7203.T": "トヨタ自動車",
    "7261.T": "マツダ", "7267.T": "ホンダ", "7272.T": "ヤマハ発動機",
    "7532.T": "パンパシHD", "7630.T": "壱番屋", "7990.T": "グローブライド",
    "8031.T": "三井物産", "8113.T": "ユニ・チャーム", "8200.T": "リンガーハット",
    "8242.T": "H2Oリテイリング", "8306.T": "三菱UFJ", "8591.T": "オリックス",
    "8593.T": "三菱HCキャピタル", "8729.T": "ソニーFH", "9041.T": "近鉄GHD",
    "9142.T": "JR九州", "9202.T": "ANAホールディングス", "9432.T": "日本電信電話",
    "9434.T": "ソフトバンク", "9828.T": "元気寿司", "9850.T": "グルメ杵屋",
    "9861.T": "吉野家HD", "9887.T": "松屋フーズ", "9936.T": "王将フード",
    "9984.T": "ソフトバンクG",
}

SECTOR_ETFS = [
    "1617.T", "1618.T", "1619.T", "1620.T", "1621.T", "1622.T",
    "1623.T", "1624.T", "1625.T", "1626.T", "1627.T", "1628.T",
    "1629.T", "1630.T", "1631.T", "1632.T", "1633.T"
]

MY_STOCKS = [
    "1326.T", "1407.T", "1419.T", "1489.T", "1605.T", "1678.T", 
    "2267.T", "2516.T", "2801.T", "2897.T", "3038.T", "3099.T", 
    "3382.T", "3397.T", "4045.T", "4543.T", "7203.T", "7261.T", 
    "7267.T", "7272.T", "7532.T", "7630.T", "7990.T", "8031.T", 
    "8113.T", "8200.T", "8242.T", "8591.T", "8593.T", "8729.T", 
    "9041.T", "9142.T", "9202.T", "9432.T", "9434.T", "9828.T", 
    "9850.T", "9861.T", "9887.T", "9936.T"
]

USER_SETTINGS = {
    "demo": MY_STOCKS, 
    "apple01": ["7203.T", "6758.T", "8306.T"],
}

# ---------------------------------------------------------
# 関数定義
# ---------------------------------------------------------
def check_login():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["user_stocks"] = []

    if st.session_state["authenticated"]:
        return st.session_state["user_stocks"]

    st.write("### 🔒 会員限定エリア")
    password_input = st.text_input("アクセスキーを入力してください", type="password")
    if st.button("ログイン"):
        if password_input in USER_SETTINGS:
            st.session_state["authenticated"] = True
            st.session_state["user_stocks"] = USER_SETTINGS[password_input]
            st.success("ログイン成功！")
            st.rerun()
        else:
            st.error("パスワードが違います")
    return None

def analyze_market(ticker_list, period, progress_bar_obj, start_progress, end_progress, get_fundamentals=False):
    results = []
    total = len(ticker_list)
    if total == 0: return []
    step = (end_progress - start_progress) / total
    
    for i, ticker in enumerate(ticker_list):
        try:
            current_bar = start_progress + (step * (i + 1))
            progress_bar_obj.progress(min(current_bar, 1.0), text=f"分析中: {ticker}")
            
            # 1. 株価データの取得
            df = yf.download(ticker, period=period, progress=False)
            
            # 2. ファンダメンタルズ情報の取得（オプション）
            per = "-"
            pbr = "-"
            yield_val = "-"
            is_good_fundamental = False

            if get_fundamentals and ticker not in SECTOR_ETFS:
                try:
                    # Tickerオブジェクトから詳細情報を取得
                    ticker_info = yf.Ticker(ticker).info
                    
                    # PER (取得できない場合はハイフン)
                    raw_per = ticker_info.get('trailingPE', None)
                    if raw_per: per = f"{raw_per:.1f}倍"
                    
                    # PBR
                    raw_pbr = ticker_info.get('priceToBook', None)
                    if raw_pbr: pbr = f"{raw_pbr:.2f}倍"

                    # 配当利回り
                    raw_yield = ticker_info.get('dividendYield', None)
                    if raw_yield: yield_val = f"{raw_yield*100:.2f}%"

                    # ★AI割安判定ロジック
                    # PER < 15 かつ PBR < 1.2 かつ 配当 > 3.0% なら「優良」
                    if (raw_per and raw_per < 15) and (raw_pbr and raw_pbr < 1.2) and (raw_yield and raw_yield > 0.03):
                        is_good_fundamental = True

                except:
                    pass

            if len(df) == 0: continue
            
            if isinstance(df.columns, pd.MultiIndex):
                 df.columns = df.columns.get_level_values(0)

            high = df['High'].max()
            low = df['Low'].min()
            current = df['Close'].iloc[-1]
            
            if high == low: pct = 0
            else: pct = ((current - low) / (high - low)) * 100

            upside = ((high - current) / current) * 100
            downside = ((current - low) / current) * 100 * -1
            
            if ticker in SECTOR_ETFS: cost_str = "-"
            else: cost_str = f"{int(current * 100):,}円"

            # ----------------------------------
            # 判定ロジック (AI総合スコア)
            # ----------------------------------
            status = "待機"
            rank = 3
            
            # 基本の底値判定
            if pct <= 10: status = "★買い"; rank = 2
            elif pct <= 20: status = "様子見"; rank = 3
            if pct >= 90: status = "⚠️高値"; rank = 5

            # ★AI特別判定（底値圏 ＋ ファンダメンタルズ良）
            if (pct <= 20) and is_good_fundamental:
                status = "👑AI推奨" # 特別なステータス
                rank = 1 # 最優先表示

            stock_name = NAME_MAP.get(ticker, "") 
            display_name = f"{ticker.replace('.T','')} {stock_name}"

            # 結果データ作成
            data_row = {
                "銘柄": display_name,
                "判定": status,
                "現在位置": f"{pct:.1f}%",
                "現在値": int(current),
                "_rank": rank,
                "_pos_val": pct
            }

            # ファンダメンタルズ列の追加
            if get_fundamentals:
                data_row["PER"] = per
                data_row["PBR"] = pbr
                data_row["配当"] = yield_val

            results.append(data_row)
        except: pass
    return results

def display_table(data_list, title, is_mobile):
    if not data_list:
        st.warning(f"{title} のデータがありません")
        return
    st.subheader(title)
    df_res = pd.DataFrame(data_list)
    df_res = df_res.sort_values(by=['_rank', '_pos_val'])
    
    # 隠し列を除外して表示用データを作る
    show_df = df_res.drop(columns=['_rank', '_pos_val'])

    # スマホ表示の列制御
    if is_mobile:
        # PERなどが含まれているかチェック
        cols = ['銘柄', '判定', '現在位置']
        if "PER" in show_df.columns:
            cols.extend(['PER', '配当']) # スマホでも重要指標は出す
        elif "現在値" in show_df.columns:
            cols.append('現在値')
            
        # 存在する列だけを表示
        existing_cols = [c for c in cols if c in show_df.columns]
        show_df = show_df[existing_cols]
    
    # 色設定（AI推奨は黄色い枠のように目立たせる）
    def highlight_row(row):
        status_val = row['判定']
        if "👑AI推奨" in status_val:
            # ゴールド（黄色）背景
            return ['background-color: #ffd700; color: black; font-weight: bold; border: 2px solid orange'] * len(row)
        elif "★買い" in status_val:
            return ['background-color: #ffcccc; color: black; font-weight: bold'] * len(row)
        elif "⚠️高値" in status_val:
            return ['background-color: #fff4cc; color: black; font-weight: bold'] * len(row)
        else:
            return [''] * len(row)

    st.dataframe(show_df.style.apply(highlight_row, axis=1), use_container_width=True, height=(len(show_df) + 1) * 35 + 3, hide_index=True)

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------

# ★ Geminiチャット
with st.expander("🤖 AIアシスタント (Gemini) に質問する"):
    if not gemini_available:
        st.error("APIキー未設定です")
    else:
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "投資の疑問にお答えします！"}]
        for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])
        if prompt := st.chat_input("例: PER10倍は割安？"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            try:
                response = model.generate_content(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                st.chat_message("assistant").write(response.text)
            except: pass

user_stocks = check_login()

if user_stocks:
    st.title("📊 日経プライム AI分析ツール")
    jst = pytz.timezone('Asia/Tokyo')
    now_str = datetime.now(jst).strftime('%Y/%m/%d %H:%M')
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"最終更新: **{now_str}**")
        period_label = st.radio("期間:", ["1年", "2年", "3年", "5年"], index=1, horizontal=True)
        selected_period = {"1年": "1y", "2年": "2y", "3年": "3y", "5年": "5y"}[period_label]
        
        # ★ファンダメンタルズ分析のON/OFFスイッチ（速度対策）
        use_fundamental = st.checkbox("詳細分析を行う（PER/PBR/配当）※少し時間がかかります", value=True)

    with col2:
        st.write("") 
        if st.button('🔄 更新'): st.rerun()

    use_mobile_view = st.toggle("📱 スマホ用シンプル表示", value=True)
    st.markdown("---")
    
    my_bar = st.progress(0, text="分析開始...")
    
    # ユーザー銘柄は詳細分析(ONの場合)
    my_results = analyze_market(user_stocks, selected_period, my_bar, 0.0, 0.7, get_fundamentals=use_fundamental)
    # ETFは詳細分析不要（False）
    sector_results = analyze_market(SECTOR_ETFS, selected_period, my_bar, 0.7, 1.0, get_fundamentals=False)
    my_bar.empty()

    display_table(my_results, "🔍 監視銘柄リスト", use_mobile_view)
    
    # AI推奨が出た場合だけ、上部に特別メッセージを出す
    top_picks = [d['銘柄'] for d in my_results if "👑AI推奨" in d['判定']]
    if top_picks:
        st.success(f"🔥 **AI激アツ判定（割安×底値×高配当）:** {'、'.join(top_picks)}")

    st.markdown("<br>", unsafe_allow_html=True)
    display_table(sector_results, "🌏 業種別トレンド (参考)", use_mobile_view)
