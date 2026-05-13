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
st.set_page_config(page_title="IA Quant - B3 & Cripto", layout="wide", initial_sidebar_state="expanded")

# CSS para ajuste de tela no tablet e botões largos
st.markdown("""
    <style>
        .main .block-container { max-width: 100%; padding-top: 0.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
        .stDataFrame { font-size: 0.75rem; }
        div.stButton > button:first-child { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# --- MAPEAMENTO DINÂMICO DE ATIVOS (SEM STOCKS EUA) ---
DICIONARIO_ATIVOS = {
    "📊 Futuros B3": ["WIN=F", "WDO=F"],
    "🚀 Criptomoedas": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "LINK-USD", "MATIC-USD"],
    "🇧🇷 Ações B3": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "MGLU3.SA", "B3SA3.SA", "RENT3.SA", "GGBR4.SA"]
}

TODOS_ATIVOS = [item for sublist in DICIONARIO_ATIVOS.values() for item in sublist][cite: 1]

# --- DIRETÓRIOS DE INTELIGÊNCIA ---
FOLDER_BRAIN = "cerebro_ia_dados"
if not os.path.exists(FOLDER_BRAIN):
    os.makedirs(FOLDER_BRAIN)[cite: 1]

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

# --- MOTOR DE APRENDIZADO NATIVO (DEEP SCAN) ---
def executar_deep_scan(selecionados):
    timeframes = {"1m": "7d", "5m": "30d", "15m": "60d", "1h": "120d"}
    progresso_sidebar = st.sidebar.progress(0)
    total_tasks = len(selecionados) * len(timeframes)
    count = 0
    
    for ativo in selecionados:
        for tf_nome, periodo in timeframes.items():
            count += 1
            path = f"{FOLDER_BRAIN}/brain_{ativo}_{tf_nome}.csv"
            try:
                hist = yf.download(ativo, period=periodo, interval=tf_nome, progress=False)
                if hist.empty or len(hist) < 50: continue
                
                # Snapshot de aprendizado persistente
                close = hist['Close'].values.flatten()
                ema200 = pd.Series(close).ewm(span=200, adjust=False).mean()
                dist = (close - ema200) / ema200
                
                estudo = [[round(dist[i], 4), 1 if close[i+5] > close[i] else 0] for i in range(50, len(close) - 5)]
                if estudo:
                    pd.DataFrame(estudo).to_csv(path, index=False, header=False)[cite: 1]
            except:
                continue
            progresso_sidebar.progress(count / total_tasks)

# --- INTERFACE LATERAL (SIDEBAR) ---
st.sidebar.title("🛰️ Radar Adaptativo")

# Seleção Dinâmica
categoria = st.sidebar.selectbox("Filtrar Categoria:", list(DICIONARIO_ATIVOS.keys()))
ativos_sugeridos = DICIONARIO_ATIVOS[categoria]

ativos_selecionados = st.sidebar.multiselect(
    "Ativos para Monitorar:", 
    options=TODOS_ATIVOS, 
    default=ativos_sugeridos[:2]
)[cite: 1]

tf_op = st.sidebar.selectbox("Timeframe de Análise", ["1m", "5m", "15m", "1h"], index=1)

if st.sidebar.button("🧠 Deep Scan (Estudar Histórico)"):
    if ativos_selecionados:
        executar_deep_scan(ativos_selecionados)
        st.sidebar.success("Memória Blindada Atualizada!")[cite: 1]
    else:
        st.sidebar.warning("Selecione ativos primeiro.")

btn_ativo = st.sidebar.toggle("🚀 Ativar Scanner em Tempo Real", value=True)

if st.sidebar.button("🗑️ Limpar Log Visual"):
    st.session_state.log_visual = []
    st.rerun()

# --- INTERFACE PRINCIPAL (LAYOUT RESTAURADO) ---
st.title("IA QUANT - LIVE MARKET")
col_sinais, col_log = st.columns([1, 1.2])[cite: 1]

if 'log_visual' not in st.session_state:
    st.session_state.log_visual = []

placeholder_sinais = col_sinais.empty()
placeholder_log = col_log.empty()

# --- LOOP DE PROCESSAMENTO ---
if btn_ativo and modelo is not None and ativos_selecionados:
    while True:
        try:
            dados = yf.download(ativos_selecionados, period="2d", interval=tf_op, progress=False, group_by='ticker')
            
            for ativo in ativos_selecionados:
                df = dados[ativo] if len(ativos_selecionados) > 1 else dados
                if df.empty or len(df) < 5: continue
                
                c = df['Close'].iloc[-1]
                ema = df['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
                tipo = "COMPRA" if c > ema else "VENDA"
                
                info = {"Ativo": ativo, "Hora": datetime.now().strftime("%H:%M:%S"), "Preço": f"{c:.2f}", "Tipo": tipo}
                
                # Registra apenas se for um novo segundo/sinal
                if not any(x['Ativo'] == ativo and x['Hora'] == info['Hora'] for x in st.session_state.log_visual):
                    st.session_state.log_visual.insert(0, info)[cite: 1]

            # Quadro de Sinais (Esquerda)
            with placeholder_sinais.container():
                st.subheader("⚡ Sinais")
                for s in st.session_state.log_visual[:6]:
                    cor = "#00FF00" if s['Tipo'] == "COMPRA" else "#FF4B4B"
                    st.markdown(f"""
                        <div style="border-left: 5px solid {cor}; padding:10px; background:#1e1e1e; border-radius:5px; margin-bottom:10px;">
                            <b style="color:white; font-size:1.1rem;">{s['Ativo']}</b><br>
                            <span style="color:{cor}; font-weight:bold;">{s['Tipo']}</span> @ {s['Preço']}
                        </div>
                    """, unsafe_allow_html=True)

            # Quadro de Log (Direita)
            with placeholder_log.container():
                st.subheader("📜 Auditoria de Sinais")
                if st.session_state.log_visual:
                    st.dataframe(pd.DataFrame(st.session_state.log_visual), use_container_width=True, hide_index=True)[cite: 1]

            time.sleep(15)
        except Exception:
            time.sleep(5)