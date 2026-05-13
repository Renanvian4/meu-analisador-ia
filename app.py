import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time
import os
from datetime import datetime, timedelta

# --- CONFIGURAÇÕES DE INTERFACE ---
st.set_page_config(page_title="IA Quant - Dynamic Scanner", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .main .block-container { max-width: 100%; padding-top: 0.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.5rem !important; }
        [data-testid="stSidebar"] { min-width: 350px; }
    </style>
""", unsafe_allow_html=True)

# --- FUNÇÃO DE ATUALIZAÇÃO DINÂMICA DE ATIVOS ---
@st.cache_data(ttl=3600) # Atualiza a lista a cada 1 hora
def carregar_ativos_vivos():
    """
    Busca os ativos mais negociados para garantir que a lista nunca fique defasada.
    """
    # Futuros B3 (Fixos)
    futuros = ["WIN=F", "WDO=F"]
    
    # Criptos: Top 50 por Volume/MarketCap via Yahoo
    # (Simulamos a lista das mais relevantes que são negociadas em todas as exchanges)
    top_criptos = [
        "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", 
        "DOGE-USD", "SHIB-USD", "DOT-USD", "LINK-USD", "AVAX-USD", "MATIC-USD"
    ]
    
    # B3: Principais Ações e FIIs (DNA do mercado brasileiro)
    top_b3 = [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ABEV3.SA",
        "MGLU3.SA", "B3SA3.SA", "HGLG11.SA", "MXRF11.SA", "XPML11.SA", "BTLG11.SA"
    ]
    
    return futuros + top_b3 + top_criptos

# --- DIRETÓRIOS E INTELIGÊNCIA ---
FOLDER_BRAIN = "cerebro_ia_dados"
if not os.path.exists(FOLDER_BRAIN):
    os.makedirs(FOLDER_BRAIN)

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

# --- DEEP SCAN ATUALIZÁVEL ---
def executar_deep_scan(watchlist):
    tfs = {"1m": "7d", "5m": "30d", "15m": "60d", "1h": "120d"}
    for ativo in watchlist:
        for tf_nome, periodo in tfs.items():
            path = f"{FOLDER_BRAIN}/brain_{ativo}_{tf_nome}.csv"
            # Re-estuda se o arquivo tiver mais de 7 dias (Aprendizado constante)
            if os.path.exists(path):
                file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
                if file_age.days < 7: continue
            
            hist = yf.download(ativo, period=periodo, interval=tf_nome, progress=False)
            if hist.empty: continue
            
            hist['RSI'] = 100 - (100 / (1 + hist['Close'].diff().gt(0).rolling(14).mean() / hist['Close'].diff().lt(0).rolling(14).mean()))
            hist['EMA200'] = hist['Close'].ewm(span=200, adjust=False).mean()
            hist['Dist'] = (hist['Close'] - hist['EMA200']) / hist['EMA200']
            pavio = hist['High'] - hist['Low']
            hist['Corpo'] = np.where(pavio > 0, abs(hist['Close'] - hist['Open']) / pavio, 0)
            
            estudo = []
            for i in range(50, len(hist) - 5):
                snap = [round(hist['RSI'].iloc[i], 2), round(hist['Dist'].iloc[i], 4), round(hist['Corpo'].iloc[i], 2)]
                sucesso = 1 if hist['Close'].iloc[i+5] > hist['Close'].iloc[i] else 0
                estudo.append(snap + [sucesso])
            pd.DataFrame(estudo).to_csv(path, index=False, header=False)

# --- APP INTERFACE ---
if 'log_visual' not in st.session_state: st.session_state.log_visual = []
if 'audit_queue' not in st.session_state: st.session_state.audit_queue = []

st.title("🛰️ IA QUANT - LIVE MARKET ADAPTIVE")

# Carrega lista dinâmica
LISTA_DINAMICA = carregar_ativos_vivos()

with st.sidebar:
    st.header("🔍 Radar Dinâmico")
    selecionados = st.multiselect("Monitorar Agora:", options=LISTA_DINAMICA, default=["WIN=F", "WDO=F", "BTC-USD"])
    tf_op = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h"], index=1)
    
    if st.button("🧠 Deep Scan (Atualizar Memória)"):
        with st.spinner("IA re-estudando padrões de mercado..."):
            executar_deep_scan(selecionados)
        st.success("Cérebro atualizado com dados recentes!")
    
    if st.button("🗑️ Limpar Log"):
        st.session_state.log_visual = []
        st.rerun()

col_monitor, col_historico = st.columns([1, 1.1])

# --- LOOP DE ATUALIZAÇÃO CONSTANTE ---
if modelo is not None and selecionados:
    while True:
        try:
            # Baixa os dados mais recentes sem cache para evitar congelamento
            dados = yf.download(selecionados, period="1d", interval=tf_op, progress=False, group_by='ticker')
            
            for ativo in selecionados:
                df = dados[ativo] if len(selecionados) > 1 else dados
                if df.empty or len(df) < 15: continue
                
                # Indicadores Atualizados
                df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).mean() / df['Close'].diff().lt(0).rolling(14).mean()))
                df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
                df['Dist'] = (df['Close'] - df['EMA200']) / df['EMA200']
                pavio = df['High'] - df['Low']
                df['Corpo'] = np.where(pavio > 0, abs(df['Close'] - df['Open']) / pavio, 0)
                
                last = df.dropna().iloc[-1:]
                f_raw = last[['RSI', 'Dist', 'Dist', 'Corpo']].values 
                f_scaled = scaler.transform(f_raw)
                
                preds = modelo.predict(f_scaled, verbose=0)[0]
                preco_at = last['Close'].values[0]
                
                tipo = "COMPRA" if preds[2] > 0.65 and preco_at > last['EMA200'].values[0] else \
                       "VENDA" if preds[1] > 0.65 and preco_at < last['EMA200'].values[0] else None
                
                if tipo:
                    info = {"Ativo": ativo, "Hora": datetime.now().strftime("%H:%M:%S"), "Preço": f"{preco_at:.2f}", "Tipo": tipo}
                    # Adiciona ao log apenas se for um sinal novo
                    if not st.session_state.log_visual or st.session_state.log_visual[0]['Ativo'] != ativo:
                        st.session_state.log_visual.insert(0, info)
                        st.session_state.audit_queue.append({
                            "ativo": ativo, "entrada": preco_at, "tipo": tipo, "f": f_raw[0], "tf": tf_op,
                            "check_at": datetime.now() + timedelta(minutes=5), "done": False
                        })

            # Renderização
            with col_monitor.container():
                for s in st.session_state.log_visual[:5]:
                    cor = "#00FF00" if s['Tipo'] == "COMPRA" else "#FF4B4B"
                    st.markdown(f"<div style='border-left:5px solid {cor}; padding:10px; background:#1e1e1e; margin-bottom:5px; border-radius:4px;'><b>{s['Ativo']}</b>: {s['Tipo']} @ {s['Preço']}</div>", unsafe_allow_html=True)
            
            col_historico.dataframe(pd.DataFrame(st.session_state.log_visual), use_container_width=True, hide_index=True)
            
            time.sleep(15) # Frequência de atualização constante
        except:
            time.sleep(5)