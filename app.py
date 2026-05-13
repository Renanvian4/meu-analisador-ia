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
        # Busca dados e calcula indicadores
        df = yf.download(ativo, period="1d", interval="1m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        
        df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).mean() / df['Close'].diff().lt(0).rolling(14).mean()))
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        df['Dist_EMA'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
        df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        df['Forca_Corpo'] = np.where((df['High']-df['Low']) > 0, abs(df['Close']-df['Open'])/(df['High']-df['Low']), 0)
        
        # Limpeza de dados para evitar valores nulos no scaler
        df_clean = df.dropna()
        
        if not df_clean.empty:
            # Predição
            features = ['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo']
            input_data = df_clean[features].iloc[-1:].values
            input_scaled = scaler.transform(input_data)
            
            # Aqui usamos 'modelo' que foi definido lá em cima
            preds = modelo.predict(input_scaled, verbose=0)[0]
            
            with placeholder.container():
                st.metric("Preço", f"{df_clean['Close'].iloc[-1]:.2f}")
                
                # Visualização de Confiança
                st.write(f"📈 Alta: {preds[2]*100:.1f}% | 📉 Baixa: {preds[1]*100:.1f}%")
                
                if preds[2] > 0.65: 
                    st.success("🚀 SINAL DE COMPRA")
                elif preds[1] > 0.65: 
                    st.error("📉 SINAL DE VENDA")
                else:
                    st.info("⏳ Monitorando padrões...")
                    
                st.line_chart(df_clean['Close'].tail(30))
        
        time.sleep(5)