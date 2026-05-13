import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time
import os
import requests
import websocket
import json
import threading
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DE INTERFACE ---
st.set_page_config(page_title="IA Quant - Omni Scanner", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .main .block-container { max-width: 100%; padding-top: 0.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
        .stDataFrame { font-size: 0.75rem; }
        .stMultiSelect div div div div { background-color: #1e1e1e !important; }
    </style>
""", unsafe_allow_html=True)

# --- GESTÃO DE DADOS WEBSOCKET (COINEXT) ---
if 'precos_cripto' not in st.session_state:
    st.session_state.precos_cripto = {}

def on_message(ws, message):
    try:
        data = json.loads(message)
        if 'Events' in data:
            for event in data['Events']:
                symbol = event.get('InstrumentId')
                price = event.get('LastTradedPrice')
                if symbol and price:
                    st.session_state.precos_cripto[str(symbol)] = price
    except:
        pass

def iniciar_websocket():
    ws = websocket.WebSocketApp("wss://api.coinext.com.br/WSGateway/", on_message=on_message)
    ws.run_forever()

if 'ws_thread' not in st.session_state:
    st.session_state.ws_thread = threading.Thread(target=iniciar_websocket, daemon=True)
    st.session_state.ws_thread.start()

# --- INTEGRAÇÃO API B3 (DADOS DE MERCADO V1) ---
@st.cache_data(ttl=86400)
def carregar_universo_b3(token):
    base_url = "https://api.dadosdemercado.com.br/v1"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r_tickers = requests.get(f"{base_url}/tickers", headers=headers)
        r_reits = requests.get(f"{base_url}/reits", headers=headers)
        lista = []
        if r_tickers.status_code == 200:
            lista += [f"{t['ticker']}.SA" for t in r_tickers.json()]
        if r_reits.status_code == 200:
            lista += [f"{r['ticker']}.SA" for r in r_reits.json()]
        return sorted(list(set(lista)))
    except:
        return ["PETR4.SA", "VALE3.SA", "ITUB4.SA"]

# --- GESTÃO DE INTELIGÊNCIA E MODELOS ---
FOLDER_BRAIN = "cerebro_ia_dados"
if not os.path.exists(FOLDER_BRAIN): os.makedirs(FOLDER_BRAIN)

@st.cache_resource
def load_assets():
    try:
        with open('scaler.pkl', 'rb') as f: s = pickle.load(f)
        m = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(4,)), 
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        m.load_weights('modelo_pesos.weights.h5')
        return m, s
    except: return None, None

modelo, scaler = load_assets()

# --- MOTOR DE DEEP SCAN MULTI-TIMEFRAME ---
def executar_deep_scan(selecionados):
    timeframes = {"1m": "7d", "5m": "30d", "15m": "60d", "1h": "120d"}
    progresso = st.sidebar.progress(0)
    total = len(selecionados) * len(timeframes)
    count = 0
    for ativo in selecionados:
        for tf_nome, periodo in timeframes.items():
            count += 1
            path = f"{FOLDER_BRAIN}/brain_{ativo}_{tf_nome}.csv"
            try:
                hist = yf.download(ativo, period=periodo, interval=tf_nome, progress=False)
                if hist.empty or len(hist) < 50: continue
                close = hist['Close'].values.flatten()
                ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
                dist = (close - ema200) / ema200
                estudo = [[round(dist[i], 4), (1 if close[i+5] > close[i] else 0)] for i in range(50, len(close)-5)]
                pd.DataFrame(estudo).to_csv(path, index=False, header=False)
            except: continue
            progresso.progress(count / total)

# --- INTERFACE E MONITORAMENTO ---
st.sidebar.title("🔍 Radar Omni-V14")
token_b3 = st.sidebar.text_input("Token B3 (V1)", type="password", value="seu_token")
universo_b3 = carregar_universo_b3(token_b3)

CRIPTO_COINEXT = ["BTC", "ETH", "SOL", "XRP", "LTC", "ADA"]
FUTUROS = ["WIN=F", "WDO=F"]
OPCOES = FUTUROS + CRIPTO_COINEXT + universo_b3

selecionados = st.sidebar.multiselect("Ativos para Rastreio:", OPCOES, default=["WIN=F", "BTC"])
tf_operacional = st.sidebar.selectbox("Timeframe Principal", ["1m", "5m", "15m", "1h"], index=1)

if st.sidebar.button("🧠 Deep Scan Massivo"):
    executar_deep_scan(selecionados)
    st.sidebar.success("Memória Blindada Atualizada!")

st.title("🛰️ IA QUANT - LIVE MARKET ADAPTIVE")
c1, c2 = st.columns([1, 1.2])

if modelo is not None and selecionados:
    placeholder_cards = c1.empty()
    placeholder_logs = c2.empty()
    if 'log_visual' not in st.session_state: st.session_state.log_visual = []

    while True:
        try:
            ativos_yf = [a for a in selecionados if ".SA" in a or "=F" in a]
            dados_yf = yf.download(ativos_yf, period="2d", interval=tf_operacional, progress=False, group_by='ticker') if ativos_yf else {}

            for ativo in selecionados:
                if ativo in CRIPTO_COINEXT:
                    preco = st.session_state.precos_cripto.get(ativo, "Aguardando...")
                    tipo = "LIVE STREAM"
                else:
                    df = dados_yf[ativo] if len(ativos_yf) > 1 else dados_yf
                    preco = round(df['Close'].iloc[-1], 2)
                    ema = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
                    tipo = "COMPRA" if preco > ema else "VENDA"

                info = {"Ativo": ativo, "Hora": datetime.now().strftime("%H:%M:%S"), "Preço": preco, "Tipo": tipo}
                if not any(x['Ativo'] == ativo and x['Hora'][:5] == info['Hora'][:5] for x in st.session_state.log_visual):
                    st.session_state.log_visual.insert(0, info)

            with placeholder_cards.container():
                for s in st.session_state.log_visual[:6]:
                    cor = "#00FF00" if "COMPRA" in str(s['Tipo']) else "#FF4B4B" if "VENDA" in str(s['Tipo']) else "#FFFF00"
                    st.markdown(f"<div style='border-left:5px solid {cor}; padding:10px; background:#1e1e1e; margin-bottom:5px; border-radius:4px;'><b>{s['Ativo']}</b>: {s['Tipo']} @ {s['Preço']}</div>", unsafe_allow_html=True)

            placeholder_logs.dataframe(pd.DataFrame(st.session_state.log_visual).head(20), use_container_width=True, hide_index=True)
            time.sleep(10)
        except:
            time.sleep(10)