import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="IA Scanner Automático", layout="wide")

@st.cache_resource
def load_assets():
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

modelo, scaler = load_assets()

# --- INICIALIZAÇÃO DE ESTADOS ---
if 'log_oportunidades' not in st.session_state:
    st.session_state.log_oportunidades = []
if 'watchlist' not in st.session_state:
    # Lista inicial padrão
    st.session_state.watchlist = ["BTC-USD", "ETH-USD", "PETR4.SA", "VALE3.SA"]

# --- FUNÇÕES DE IMPORTAÇÃO AUTOMÁTICA ---
def importar_top_criptos():
    # Lista das 20 principais moedas (Simulando CoinMarketCap via Yahoo)
    top_coins = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "SHIB-USD", "AVAX-USD", "LINK-USD", "BCH-USD", "UNI-USD", "XLM-USD", "LEO-USD", "ETC-USD", "ATOM-USD"]
    for coin in top_coins:
        if coin not in st.session_state.watchlist:
            st.session_state.watchlist.append(coin)

def importar_bovespa():
    # Lista das principais ações da B3 (Blue Chips)
    b3_stocks = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "BBAS3.SA", "B3SA3.SA", "ITSA4.SA", "MGLU3.SA", "HAPV3.SA", "RENT3.SA", "JBSS3.SA", "SUZB3.SA", "WEGE3.SA", "GGBR4.SA", "CSNA3.SA", "LREN3.SA", "PRIO3.SA"]
    for stock in b3_stocks:
        if stock not in st.session_state.watchlist:
            st.session_state.watchlist.append(stock)

# --- BARRA LATERAL ---
st.sidebar.header("🔍 Radar de Ativos")

# Botões de Importação em Massa
col_bt1, col_bt2 = st.sidebar.columns(2)
if col_bt1.button("🌐 Top Criptos"):
    importar_top_criptos()
if col_bt2.button("🇧🇷 Bovespa"):
    importar_bovespa()

# Campo Manual (Corrigido)
novo_ticker = st.sidebar.text_input("Adicionar ticker manual (ex: SOL-USD):").upper()
if st.sidebar.button("➕ Adicionar"):
    if novo_ticker and novo_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(novo_ticker)
        st.rerun()

# Multiselect que reflete o estado real da watchlist
ativos_final = st.sidebar.multiselect(
    "Ativos sendo monitorados:",
    options=st.session_state.watchlist,
    default=st.session_state.watchlist
)

if st.sidebar.button("🗑️ Limpar Lista"):
    st.session_state.watchlist = []
    st.rerun()

# --- INTERFACE PRINCIPAL ---
st.title("🛰️ Scanner IA Multi-Mercado")
col_monitor, col_log = st.columns([2, 1])

with col_monitor:
    st.subheader("⚡ Sinais Ativos")
    placeholder_cards = st.empty()

with col_log:
    st.subheader("📜 Histórico de Sinais")
    placeholder_log = st.empty()

# --- LOOP DE PROCESSAMENTO ---
if st.sidebar.toggle('▶️ Ligar Radar IA'):
    if not ativos_final:
        st.warning("Adicione ativos para começar a varredura.")
    else:
        while True:
            try:
                # Download em massa otimizado
                dados = yf.download(ativos_final, period="7d", interval="5m", progress=False, group_by='ticker')
                
                oportunidades = []
                
                for ativo in ativos_final:
                    df = dados[ativo] if len(ativos_final) > 1 else dados
                    if df.empty or len(df) < 50: continue
                    
                    # Cálculo de Indicadores
                    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).mean() / df['Close'].diff().lt(0).rolling(14).mean()))
                    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
                    df['Dist_EMA'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
                    df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
                    pavio = df['High'] - df['Low']
                    df['Forca_Corpo'] = np.where(pavio > 0, abs(df['Close'] - df['Open']) / pavio, 0)
                    
                    df_clean = df.dropna(subset=['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo'])
                    
                    if not df_clean.empty:
                        input_scaled = scaler.transform(df_clean[['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo']].iloc[-1:].values)
                        preds = modelo.predict(input_scaled, verbose=0)[0]
                        
                        tipo = "COMPRA" if preds[2] > 0.65 else "VENDA" if preds[1] > 0.65 else None
                        
                        if tipo:
                            info = {
                                "Ativo": ativo,
                                "Hora": datetime.now().strftime("%H:%M:%S"),
                                "Preço": f"{df_clean['Close'].iloc[-1]:.2f}",
                                "Tipo": tipo,
                                "Confiança": f"{max(preds[1], preds[2])*100:.1f}%"
                            }
                            oportunidades.append(info)
                            
                            # Log único: Só adiciona se o último registro do ativo for diferente ou tiver mais de 5 min
                            if not st.session_state.log_oportunidades or st.session_state.log_oportunidades[0]['Ativo'] != ativo:
                                st.session_state.log_oportunidades.insert(0, info)

                # --- RENDERIZAÇÃO ---
                with placeholder_cards.container():
                    if oportunidades:
                        # Grid dinâmico de cards
                        for i in range(0, len(oportunidades), 4):
                            cols = st.columns(4)
                            for j, op in enumerate(oportunidades[i:i+4]):
                                cor = "#00FF00" if op['Tipo'] == "COMPRA" else "#FF4B4B"
                                with cols[j]:
                                    st.markdown(f"""
                                        <div style="border:2px solid {cor}; padding:10px; border-radius:10px; background:#1e1e1e; text-align:center; margin-bottom:10px;">
                                            <h4 style="margin:0;">{op['Ativo']}</h4>
                                            <h2 style="color:{cor}; margin:5px 0;">{op['Tipo']}</h2>
                                            <p style="margin:0; font-weight:bold;">{op['Preço']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("🔎 Escaneando mercado... Sem sinais de alta probabilidade.")

                with placeholder_log.container():
                    if st.session_state.log_oportunidades:
                        st.table(pd.DataFrame(st.session_state.log_oportunidades).head(20))

                time.sleep(10) # Tempo seguro para monitorar muitos ativos

            except Exception as e:
                st.error(f"Erro na varredura: {e}")
                time.sleep(5)