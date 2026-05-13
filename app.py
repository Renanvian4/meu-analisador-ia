import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time

# 1. Carregar a IA
@st.cache_resource
def load_assets():
    # 1. Carrega o Scaler
    with open('scaler.pkl', 'rb') as f:
        s = pickle.load(f)
    
    # 2. Reconstrói a arquitetura exatamente como no treino
    m = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(4,)), 
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    # 3. Carrega apenas os pesos
    m.load_weights('modelo_pesos.weights.h5')
    return m, s

# Inicializa as variáveis globais do modelo
modelo, scaler = load_assets()

# 2. Interface
st.title("🤖 IA Trader Mobile")
ativo = st.selectbox("Par do TradingView", ["BTC-USD", "ETH-USD", "SOL-USD", "PETR4.SA", "VALE3.SA"])

if st.button('Ligar Analisador'):
    placeholder = st.empty()
    while True:
        # Reduzimos o período de busca para 5 dias (mínimo seguro para calcular EMA 200 em 1m)
        # O argumento progress=False evita poluir os logs e acelera a execução
        df = yf.download(ativo, period="5d", interval="1m", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        
        # ... (mantém os cálculos de indicadores iguais) ...
        
        df_clean = df.dropna()
        
        if not df_clean.empty:
            features = ['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo']
            input_data = df_clean[features].iloc[-1:].values
            input_scaled = scaler.transform(input_data)
            
            preds = modelo.predict(input_scaled, verbose=0)[0]
            
            with placeholder.container():
                # Interface mais compacta para facilitar o refresh visual
                st.subheader(f"📊 {ativo} - {time.strftime('%H:%M:%S')}")
                st.metric("Preço", f"{df_clean['Close'].iloc[-1]:.2f}")
                
                # ... (mantém a lógica de sucesso/erro dos sinais) ...
                
            # --- O PONTO CHAVE: REDUÇÃO DO TEMPO ---
            # 2 segundos é o limite seguro para evitar erro 429 (Too Many Requests)
            time.sleep(2) 
