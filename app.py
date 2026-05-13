import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import time
from datetime import datetime

# --- CONFIGURAÇÃO DE INTERFACE ---
st.set_page_config(page_title="IA Quant - Real Time", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .main .block-container { max-width: 100%; padding-top: 0.5rem; }
        .stDataFrame { font-size: 0.75rem; }
        div.stButton > button:first-child { width: 100%; }
        .signal-card { border: 1px solid #444; padding: 10px; border-radius: 8px; text-align: center; background-color: #1e1e1e; margin-bottom:10px; }
    </style>
""", unsafe_allow_html=True)

# --- DIRETÓRIOS E ASSETS ---
FOLDER_BRAIN = "cerebro_ia_dados"
if not os.path.exists(FOLDER_BRAIN):
    os.makedirs(FOLDER_BRAIN)[cite: 5]

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

modelo, scaler = load_assets()[cite: 5]

# --- SIDEBAR (CONTROLES) ---
st.sidebar.title("🛰️ Radar Adaptativo")

DICIONARIO_BASE = {
    "🚀 Cripto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"],
    "📊 B3": ["WIN=F", "WDO=F", "PETR4.SA", "VALE3.SA"]
}

cat = st.sidebar.selectbox("Filtrar Categoria:", list(DICIONARIO_BASE.keys()))
ativos_favoritos = st.sidebar.multiselect("Favoritos:", DICIONARIO_BASE[cat], default=DICIONARIO_BASE[cat][:2])
busca_extra = st.sidebar.text_input("Incluir Ativo Extra:").upper().strip()

watchlist = list(ativos_favoritos)
if busca_extra and busca_extra not in watchlist:
    watchlist.append(busca_extra)[cite: 8]

tf_op = st.sidebar.selectbox("Timeframe:", ["1m", "5m", "15m", "1h"], index=0)

if st.sidebar.button("🧠 Deep Scan"):
    prog = st.sidebar.progress(0)
    for i, ativo in enumerate(watchlist):
        try:
            yf.download(ativo, period="30d", interval=tf_op, progress=False).to_csv(f"{FOLDER_BRAIN}/{ativo}.csv")
        except: pass
        prog.progress((i + 1) / len(watchlist))
    st.sidebar.success("Memória IA Atualizada!")[cite: 8, 9]

btn_on = st.sidebar.toggle("🚀 Iniciar Scanner IA", value=False)

if st.sidebar.button("🗑️ Limpar Histórico"):
    st.session_state.log_visual = []
    st.rerun()

# --- LAYOUT PRINCIPAL ---
st.title("🛰️ IA QUANT - LIVE MARKET")
col_sinais, col_log = st.columns([1, 1.2])

if 'log_visual' not in st.session_state:
    st.session_state.log_visual = [][cite: 9]

# --- RENDERIZAÇÃO E PROCESSAMENTO ---
with col_sinais:
    st.subheader("⚡ Sinais Ativos")
    area_sinais = st.empty()

with col_log:
    st.subheader("📜 Auditoria")
    area_log = st.empty()

if not btn_on:
    area_sinais.warning("Scanner Pausado.")
    area_log.info("Aguardando ativação.")
else:
    # Loop de alta frequência[cite: 9, 13, 18]
    while True:
        try:
            for ativo in watchlist:
                df = yf.download(ativo, period="5d", interval=tf_op, progress=False)
                if df.empty or len(df) < 50: continue
                if isinstance(df.columns, pd.MultiIndex): 
                    df.columns = df.columns.get_level_values(0)
                
                # Cálculos IA
                df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).mean() / df['Close'].diff().lt(0).rolling(14).mean()))
                df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
                df['Dist_EMA'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
                df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
                pavio = df['High'] - df['Low']
                df['Forca_Corpo'] = np.where(pavio > 0, abs(df['Close'] - df['Open']) / pavio, 0)
                
                df_clean = df.dropna(subset=['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo'])
                
                if not df_clean.empty and modelo is not None:
                    features = ['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo']
                    input_scaled = scaler.transform(df_clean[features].iloc[-1:].values)
                    preds = modelo.predict(input_scaled, verbose=0)[0]
                    
                    prob_venda, prob_compra = preds[1], preds[2]
                    status, cor = "⏳ Neutro", "white"
                    
                    if prob_compra > 0.52: status, cor = "🚀 COMPRA", "#00FF00"
                    elif prob_venda > 0.52: status, cor = "📉 VENDA", "#FF4B4B"
                    
                    if status != "⏳ Neutro":
                        info = {
                            "Ativo": ativo, "Hora": datetime.now().strftime("%H:%M:%S"), 
                            "Preço": f"{df_clean['Close'].iloc[-1]:.2f}", 
                            "Status": status, "Confiança": f"{max(prob_compra, prob_venda)*100:.1f}%", "Color": cor
                        }
                        if not any(x['Ativo'] == ativo and x['Hora'][:5] == info['Hora'][:5] for x in st.session_state.log_visual):
                            st.session_state.log_visual.insert(0, info)

            # Atualização instantânea dos containers[cite: 9, 14, 18]
            with area_sinais.container():
                for s in st.session_state.log_visual[:8]:
                    st.markdown(f'<div class="signal-card"><b style="font-size:1.1rem;">{s["Ativo"]}</b><br><span style="color:{s["Color"]}; font-weight:bold;">{s["Status"]}</span> | {s["Preço"]}</div>', unsafe_allow_html=True)

            with area_log.container():
                if st.session_state.log_visual:
                    st.table(pd.DataFrame(st.session_state.log_visual).drop(columns=['Color']))[cite: 9]
            
            time.sleep(2) # COOLDOWN REDUZIDO PARA 2 SEGUNDOS
        except Exception:
            time.sleep(2)