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
st.set_page_config(page_title="IA Quant - Live Market", layout="wide", initial_sidebar_state="expanded")

# CSS para ajuste no Tablet
st.markdown("""
    <style>
        .main .block-container { max-width: 100%; padding-top: 0.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
        .stDataFrame { font-size: 0.75rem; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
FOLDER_BRAIN = "cerebro_ia_dados"
if not os.path.exists(FOLDER_BRAIN):
    os.makedirs(FOLDER_BRAIN)

# --- CARREGAMENTO DE MODELOS ---
@st.cache_resource
def load_assets():
    try:
        with open('scaler.pkl', 'rb') as f:
            s = pickle.load(f)
        # Reconstrução leve do modelo
        m = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(4,)), 
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        m.load_weights('modelo_pesos.weights.h5')
        return m, s
    except Exception:
        return None, None

modelo, scaler = load_assets()

# --- MOTOR DE APRENDIZADO NATIVO (DEEP SCAN) ---
def executar_deep_scan(selecionados):
    # Timeframes mais usados por traders
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
                
                # Transformação em vetores numéricos compactos para o tablet
                close = hist['Close'].values.flatten()
                
                # Indicadores calculados nativamente
                delta = pd.Series(close).diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rsi = 100 - (100 / (1 + (gain / loss)))
                ema200 = pd.Series(close).ewm(span=200, adjust=False).mean()
                dist = (close - ema200) / ema200
                
                estudo = []
                for i in range(50, len(close) - 5):
                    if np.isnan(rsi[i]): continue
                    # Snapshot: RSI, Distância EMA, Resultado futuro (1=subiu, 0=caiu)
                    snap = [round(rsi[i], 2), round(dist[i], 4), 1 if close[i+5] > close[i] else 0]
                    estudo.append(snap)
                
                if estudo:
                    pd.DataFrame(estudo).to_csv(path, index=False, header=False)
            except:
                continue
            progresso.progress(count / total)

# --- INTERFACE E MONITORAMENTO ---
if 'log_visual' not in st.session_state:
    st.session_state.log_visual = []

st.sidebar.title("🔍 Radar Dinâmico")
# Lista expandida incluindo Futuros, B3 e Internacionais
opcoes = ["WIN=F", "WDO=F", "BTC-USD", "ETH-USD", "PETR4.SA", "VALE3.SA", "AAPL", "NVDA"]
selecionados = st.sidebar.multiselect("Monitorar Agora:", opcoes, default=["WIN=F", "WDO=F"])
tf_operacional = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=1)

if st.sidebar.button("🧠 Deep Scan (Aprendizado Nativo)"):
    executar_deep_scan(selecionados)
    st.sidebar.success("Memória de Longo Prazo Atualizada!")

st.title("🛰️ IA QUANT - LIVE MARKET ADAPTIVE")
c1, c2 = st.columns([1, 1.2])

if modelo is not None and selecionados:
    while True:
        try:
            dados = yf.download(selecionados, period="2d", interval=tf_operacional, progress=False, group_by='ticker')
            for ativo in selecionados:
                df = dados[ativo] if len(selecionados) > 1 else dados
                if df.empty or len(df) < 30: continue
                
                c = df['Close'].iloc[-1]
                ema = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
                
                # Lógica de Tendência Corrigida
                tipo = "COMPRA" if c > ema else "VENDA"
                
                info = {"Ativo": ativo, "Hora": datetime.now().strftime("%H:%M"), "Preço": f"{c:.2f}", "Tipo": tipo}
                if not any(x['Ativo'] == ativo and x['Hora'] == info['Hora'] for x in st.session_state.log_visual):
                    st.session_state.log_visual.insert(0, info)
            
            c1.subheader("⚡ Sinais")
            for s in st.session_state.log_visual[:5]:
                cor = "#00FF00" if s['Tipo'] == "COMPRA" else "#FF4B4B"
                c1.markdown(f"<div style='border-left:4px solid {cor}; padding:10px; background:#1e1e1e; margin-bottom:5px;'><b>{s['Ativo']}</b>: {s['Tipo']} @ {s['Preço']}</div>", unsafe_allow_html=True)
            
            c2.subheader("📜 Histórico")
            c2.dataframe(pd.DataFrame(st.session_state.log_visual), use_container_width=True)
            time.sleep(20)
        except:
            time.sleep(10)