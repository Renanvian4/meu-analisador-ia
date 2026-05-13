import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time
import os
from datetime import datetime, timedelta

# --- CONFIGURAÇÃO DE INTERFACE ---
st.set_page_config(page_title="IA Analitico - Multi-Ativos", layout="wide", initial_sidebar_state="expanded")

# Estilização para tablet
st.markdown("""
    <style>
        .main .block-container { max-width: 100%; padding-top: 0.5rem; }
        .stDataFrame { font-size: 0.75rem; }
        div.stButton > button:first-child { width: 100%; }
        .signal-card { border-left: 5px solid #00FF00; padding:10px; background:#1e1e1e; border-radius:5px; margin-bottom:10px; }
    </style>
""", unsafe_allow_html=True)

# --- MAPEAMENTO DE ATIVOS BASE ---
DICIONARIO_BASE = {
    "📊 Futuros B3": ["WIN=F", "WDO=F"],
    "🚀 Cripto (Favoritas)": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"],
    "🇧🇷 Ações B3": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"]
}

# --- DIRETÓRIOS ---
FOLDER_BRAIN = "cerebro_ia_dados"
if not os.path.exists(FOLDER_BRAIN):
    os.makedirs(FOLDER_BRAIN)

# --- CARREGAMENTO DE MODELOS ---
@st.cache_resource
def load_assets():
    try:
        with open('scaler.pkl', 'rb') as f:
            s = pickle.load(f)
        m = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(4,)), 
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        m.load_weights('modelo_pesos.weights.h5')
        return m, s
    except:
        return None, None

modelo, scaler = load_assets()

# --- MOTOR DE DEEP SCAN ---
def executar_deep_scan(selecionados):
    timeframes = {"1m": "7d", "5m": "30d", "15m": "60d", "1h": "120d"}
    prog = st.sidebar.progress(0)
    total = len(selecionados) * len(timeframes)
    count = 0
    for ativo in selecionados:
        for tf_nome, periodo in timeframes.items():
            count += 1
            try:
                hist = yf.download(ativo, period=periodo, interval=tf_nome, progress=False)
                if not hist.empty:
                    close = hist['Close'].values.flatten()
                    ema = pd.Series(close).ewm(span=200, adjust=False).mean()
                    dist = (close - ema) / ema
                    df_p = pd.DataFrame([[dist[i], 1 if close[i+5] > close[i] else 0] for i in range(50, len(close)-5)])
                    df_p.to_csv(f"{FOLDER_BRAIN}/brain_{ativo}_{tf_nome}.csv", index=False, header=False)
            except: pass
            prog.progress(count / total)

# --- SIDEBAR (BUSCA LIVRE DE CRIPTOS) ---
st.sidebar.title("🛰️ Radar Adaptativo")

cat = st.sidebar.selectbox("Filtrar Categoria:", list(DICIONARIO_BASE.keys()))
ativos_favoritos = st.sidebar.multiselect("Favoritos:", DICIONARIO_BASE[cat], default=DICIONARIO_BASE[cat][:2])

st.sidebar.markdown("---")
# Campo para digitar QUALQUER cripto (ex: ADA-USD, SHIB-USD, AVAX-USD)
busca_extra = st.sidebar.text_input("Incluir Cripto Extra (Ticker API):").upper().strip()

# Consolidação da lista de monitoramento
ativos_sel = list(ativos_favoritos)
if busca_extra:
    if busca_extra not in ativos_sel:
        ativos_sel.append(busca_extra)

tf_op = st.sidebar.selectbox("Timeframe:", ["1m", "5m", "15m", "1h"], index=1)

if st.sidebar.button("🧠 Deep Scan (Estudar Selecionados)"):
    if ativos_sel:
        executar_deep_scan(ativos_sel)
        st.sidebar.success("Memória Atualizada!")

btn_on = st.sidebar.toggle("🚀 Scanner em Tempo Real", value=True)
if st.sidebar.button("🗑️ Limpar Log Visual"):
    st.session_state.log_visual = []
    st.rerun()

# --- LAYOUT PRINCIPAL ---
st.title("IA QUANT - LIVE MARKET")
col_sinais, col_log = st.columns([1, 1.2])

if 'log_visual' not in st.session_state:
    st.session_state.log_visual = []

with col_sinais:
    st.subheader("⚡ Sinais")
    area_sinais = st.empty()

with col_log:
    st.subheader("📜 Auditoria")
    area_log = st.empty()

# --- LOOP DE PROCESSAMENTO ---
if btn_on and modelo is not None and ativos_sel:
    while True:
        try:
            dados = yf.download(ativos_sel, period="2d", interval=tf_op, progress=False, group_by='ticker')
            
            for ativo in ativos_sel:
                df = dados[ativo] if len(ativos_sel) > 1 else dados
                if df.empty or len(df) < 5: continue
                
                c = df['Close'].iloc[-1]
                ema = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
                tipo = "COMPRA" if c > ema else "VENDA"
                
                info = {"Ativo": ativo, "Hora": datetime.now().strftime("%H:%M:%S"), "Preço": f"{c:.2f}", "Tipo": tipo}
                
                if not any(x['Ativo'] == ativo and x['Hora'] == info['Hora'] for x in st.session_state.log_visual):
                    st.session_state.log_visual.insert(0, info)

            with area_sinais.container():
                for s in st.session_state.log_visual[:6]:
                    cor = "#00FF00" if s['Tipo'] == "COMPRA" else "#FF4B4B"
                    st.markdown(f'<div class="signal-card" style="border-left-color:{cor}"><b>{s["Ativo"]}</b><br><span style="color:{cor}">{s["Tipo"]}</span> @ {s["Preço"]}</div>', unsafe_allow_html=True)

            with area_log.container():
                if st.session_state.log_visual:
                    st.dataframe(pd.DataFrame(st.session_state.log_visual), use_container_width=True, hide_index=True)

            time.sleep(10)
        except:
            time.sleep(5)