import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time
import os
from datetime import datetime

# --- CONFIGURAÇÃO DE INTERFACE ---
st.set_page_config(page_title="IA Quant - Radar Adaptativo", layout="wide", initial_sidebar_state="expanded")

# Estilização para tablet[cite: 7]
st.markdown("""
    <style>
        .main .block-container { max-width: 100%; padding-top: 0.5rem; }
        .stDataFrame { font-size: 0.75rem; }
        div.stButton > button:first-child { width: 100%; }
        .signal-card { border: 1px solid #444; padding: 10px; border-radius: 10px; text-align: center; background-color: #1e1e1e; margin-bottom:10px; }
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
FOLDER_BRAIN = "cerebro_ia_dados"
if not os.path.exists(FOLDER_BRAIN):
    os.makedirs(FOLDER_BRAIN)

# --- CARREGAMENTO DE MODELO E SCALER ---
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

# --- FUNÇÃO DE APRENDIZADO (DEEP SCAN) ---
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
                if hist.empty: continue
                # Limpeza de colunas MultiIndex
                if isinstance(hist.columns, pd.MultiIndex): hist.columns = hist.columns.get_level_values(0)
                
                # Cálculos para a memória da IA
                close = hist['Close'].values.flatten()
                ema = pd.Series(close).ewm(span=200, adjust=False).mean()
                dist = (close - ema) / ema
                df_p = pd.DataFrame([[dist[i], 1 if close[i+5] > close[i] else 0] for i in range(50, len(close)-5)])
                df_p.to_csv(f"{FOLDER_BRAIN}/brain_{ativo}_{tf_nome}.csv", index=False, header=False)
            except: pass
            prog.progress(count / total)

# --- SIDEBAR (INTERFACE DE CONTROLE) ---
st.sidebar.title("🛰️ Radar Adaptativo")

DICIONARIO_BASE = {
    "🚀 Cripto (Favoritas)": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"],
    "📊 Futuros B3": ["WIN=F", "WDO=F"],
    "🇧🇷 Ações B3": ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"]
}

cat = st.sidebar.selectbox("Filtrar Categoria:", list(DICIONARIO_BASE.keys()))
favoritos = st.sidebar.multiselect("Favoritos:", DICIONARIO_BASE[cat], default=DICIONARIO_BASE[cat][:2])

# CAMPO RESTAURADO: Busca Livre para incluir novos ativos
busca_extra = st.sidebar.text_input("Incluir Ativo Extra (Ticker API):").upper().strip()

# Consolidação da lista de monitoramento
ativos_sel = list(favoritos)
if busca_extra and busca_extra not in ativos_sel:
    ativos_sel.append(busca_extra)

tf_op = st.sidebar.selectbox("Timeframe de Operação:", ["1m", "5m", "15m", "1h"], index=1)

# FUNÇÃO RESTAURADA: Botão de Deep Scan
if st.sidebar.button("🧠 Deep Scan (Aprendizado Profundo)"):
    if ativos_sel:
        executar_deep_scan(ativos_sel)
        st.sidebar.success("Memória IA Atualizada!")

# Scanner inicia desativado conforme solicitado
btn_on = st.sidebar.toggle("🚀 Iniciar Scanner IA", value=False)

if st.sidebar.button("🗑️ Limpar Histórico"):
    st.session_state.log_visual = []
    st.rerun()

# --- QUADRO PRINCIPAL ---
st.title("🛰️ IA QUANT - LIVE MARKET ADAPTIVE")
col_sinais, col_log = st.columns([1, 1.2])

if 'log_visual' not in st.session_state:
    st.session_state.log_visual = []

# containers fixos para exibição central e lateral
with col_sinais:
    st.subheader("⚡ Sinais Ativos")
    area_sinais = st.empty()

with col_log:
    st.subheader("📜 Auditoria de Sinais")
    area_log = st.empty()

# --- LOOP DE PROCESSAMENTO (IA REAL TIME) ---
if btn_on and modelo is not None and ativos_sel:
    while True:
        try:
            for ativo in ativos_sel:
                # CORREÇÃO: period="7d" garante dados para EMA200[cite: 7]
                df = yf.download(ativo, period="7d", interval=tf_op, progress=False)
                if df.empty or len(df) < 50: continue
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                
                # CÁLCULO DOS 4 INDICADORES DA IA
                df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).mean() / df['Close'].diff().lt(0).rolling(14).mean()))
                df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
                df['Dist_EMA'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
                df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
                pavio = df['High'] - df['Low']
                df['Forca_Corpo'] = np.where(pavio > 0, abs(df['Close'] - df['Open']) / pavio, 0)
                
                df_clean = df.dropna(subset=['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo'])
                
                if not df_clean.empty:
                    # Predição Real com o Modelo .h5[cite: 7]
                    features = ['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo']
                    input_scaled = scaler.transform(df_clean[features].iloc[-1:].values)
                    preds = modelo.predict(input_scaled, verbose=0)[0]
                    prob_neutro, prob_venda, prob_compra = preds
                    
                    status, cor = "⏳ Neutro", "white"
                    if prob_compra > 0.60:
                        status, cor = "🚀 COMPRA", "#00FF00"
                    elif prob_venda > 0.60:
                        status, cor = "📉 VENDA", "#FF4B4B"
                    
                    if status != "⏳ Neutro":
                        info = {
                            "Ativo": ativo, 
                            "Hora": datetime.now().strftime("%H:%M:%S"), 
                            "Preço": f"{df_clean['Close'].iloc[-1]:.2f}", 
                            "Status": status,
                            "Confiança": f"{max(prob_compra, prob_venda)*100:.1f}%",
                            "Color": cor
                        }
                        # Adiciona se for sinal novo no minuto
                        if not any(x['Ativo'] == ativo and x['Hora'][:5] == info['Hora'][:5] for x in st.session_state.log_visual):
                            st.session_state.log_visual.insert(0, info)

            # ATUALIZAÇÃO VISUAL IMEDIATA
            with area_sinais.container():
                for s in st.session_state.log_visual[:6]:
                    st.markdown(f"""
                        <div class="signal-card">
                            <h3 style="margin:0;">{s['Ativo']}</h3>
                            <h2 style="color:{s['Color']}; margin:5px 0;">{s['Status']}</h2>
                            <p style="margin:0;">Preço: {s['Preço']} | {s['Confiança']}</p>
                        </div>
                    """, unsafe_allow_html=True)

            with area_log.container():
                if st.session_state.log_visual:
                    st.dataframe(pd.DataFrame(st.session_state.log_visual).drop(columns=['Color']), use_container_width=True, hide_index=True)

            time.sleep(10)
        except:
            time.sleep(5)
elif not btn_on:
    area_sinais.warning("Scanner Pausado. Ative na lateral para ver os sinais.")