import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="IA Scanner Dinâmico", layout="wide")

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

# Inicialização de estados
if 'log_oportunidades' not in st.session_state:
    st.session_state.log_oportunidades = []

# --- BARRA LATERAL PARA INCLUSÃO DE ATIVOS ---
st.sidebar.header("⚙️ Gestão de Ativos")

# Sugestões iniciais para facilitar
sugestoes = ["BTC-USD", "ETH-USD", "SOL-USD", "PETR4.SA", "VALE3.SA", "ITUB4.SA", "AAPL", "NVDA"]

# ESTE É O CAMPO QUE REALMENTE ADICIONA ATIVOS
# Você pode digitar qualquer ticker e dar 'Enter'
ativos_escolhidos = st.sidebar.multiselect(
    "Adicione ou remova ativos:",
    options=sugestoes + st.session_state.get('custom_tickers', []),
    default=["BTC-USD", "ETH-USD", "PETR4.SA"]
)

# Campo para digitar um ticker que não está na lista de sugestões
novo_ticker = st.sidebar.text_input("Digitar novo ticker (ex: MGLU3.SA, DOGE-USD):").upper()
if st.sidebar.button("Adicionar à Lista"):
    if novo_ticker and novo_ticker not in sugestoes:
        if 'custom_tickers' not in st.session_state:
            st.session_state.custom_tickers = []
        st.session_state.custom_tickers.append(novo_ticker)
        st.rerun()

st.sidebar.write(f"Total de ativos: {len(ativos_escolhidos)}")

# --- LAYOUT PRINCIPAL ---
st.title("🛰️ Scanner IA em Tempo Real")
col_monitor, col_log = st.columns([2, 1])

with col_monitor:
    st.subheader("⚡ Sinais em Evidência")
    placeholder_cards = st.empty()

with col_log:
    st.subheader("📜 Log de Oportunidades")
    placeholder_log = st.empty()

# --- LOOP DE VARREDURA ---
if st.sidebar.toggle('▶️ Iniciar Monitoramento'):
    if not ativos_escolhidos:
        st.warning("Adicione pelo menos um ativo para iniciar.")
    else:
        while True:
            try:
                # Download em lote dos ativos selecionados no campo multiselect
                dados_brutos = yf.download(ativos_escolhidos, period="7d", interval="5m", progress=False, group_by='ticker')
                
                oportunidades_atuais = []
                
                for ativo in ativos_escolhidos:
                    # Tratamento para download de ativo único ou múltiplos
                    df = dados_brutos[ativo] if len(ativos_escolhidos) > 1 else dados_brutos
                    
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
                        features = ['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo']
                        input_scaled = scaler.transform(df_clean[features].iloc[-1:].values)
                        preds = modelo.predict(input_scaled, verbose=0)[0]
                        
                        tipo = None
                        if preds[2] > 0.65: tipo = "COMPRA"
                        elif preds[1] > 0.65: tipo = "VENDA"
                        
                        if tipo:
                            info = {
                                "Ativo": ativo,
                                "Hora": datetime.now().strftime("%H:%M:%S"),
                                "Preço": f"{df_clean['Close'].iloc[-1]:.2f}",
                                "Tipo": tipo,
                                "Confiança": f"{max(preds[1], preds[2])*100:.1f}%"
                            }
                            oportunidades_atuais.append(info)
                            
                            # Registra no Log se for novo
                            if not st.session_state.log_oportunidades or st.session_state.log_oportunidades[0]['Ativo'] != ativo or st.session_state.log_oportunidades[0]['Tipo'] != tipo:
                                st.session_state.log_oportunidades.insert(0, info)

                # --- RENDERIZAÇÃO ---
                with placeholder_cards.container():
                    if oportunidades_atuais:
                        # Exibe até 8 cards em grid 4x2
                        for i in range(0, len(oportunidades_atuais), 4):
                            cols = st.columns(4)
                            for j, op in enumerate(oportunidades_atuais[i:i+4]):
                                cor = "#00FF00" if op['Tipo'] == "COMPRA" else "#FF4B4B"
                                with cols[j]:
                                    st.markdown(f"""
                                        <div style="border:2px solid {cor}; padding:10px; border-radius:10px; background:#1e1e1e; text-align:center;">
                                            <h4 style="margin:0;">{op['Ativo']}</h4>
                                            <h3 style="color:{cor}; margin:2px 0;">{op['Tipo']}</h3>
                                            <p style="margin:0; font-weight:bold;">{op['Preço']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.info("🔎 Analisando lista de ativos... Sem sinais claros.")

                with placeholder_log.container():
                    if st.session_state.log_oportunidades:
                        st.table(pd.DataFrame(st.session_state.log_oportunidades).head(15))

                time.sleep(5) # Delay otimizado

            except Exception as e:
                st.sidebar.error(f"Erro de conexão: {e}")
                time.sleep(5)