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
        # Aumentamos para '7d' para garantir que SEMPRE haja 200 candles de 1m para a EMA
        df = yf.download(ativo, period="7d", interval="1m", progress=False)
        
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)
        
        # --- CÁLCULOS COM PROTEÇÃO ---
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain/loss))
        
        # EMA 200 e Distância
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        df['Dist_EMA'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
        
        # Volume e Volatilidade
        df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        
        pavio = df['High'] - df['Low']
        df['Forca_Corpo'] = np.where(pavio > 0, abs(df['Close'] - df['Open']) / pavio, 0)
        
        # --- LIMPEZA E VALIDAÇÃO ---
        features = ['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo']
        
        # Removemos linhas nulas que surgem no início dos cálculos das médias
        df_clean = df.dropna(subset=features)
        
        if not df_clean.empty:
            # Selecionamos os dados para a IA
            input_data = df_clean[features].iloc[-1:].values
            input_scaled = scaler.transform(input_data)
            
            preds = modelo.predict(input_scaled, verbose=0)[0]
            
            with placeholder.container():
                st.subheader(f"📊 {ativo} - {time.strftime('%H:%M:%S')}")
                st.metric("Preço", f"{df_clean['Close'].iloc[-1]:.2f}")
                
                # Sinais Visuais
                c1, c2 = st.columns(2)
                c1.write(f"📈 Alta: {preds[2]*100:.1f}%")
                c2.write(f"📉 Baixa: {preds[1]*100:.1f}%")
                
                if preds[2] > 0.60: st.success("🚀 SINAL DE COMPRA")
                elif preds[1] > 0.60: st.error("📉 SINAL DE VENDA")
                
                st.line_chart(df_clean['Close'].tail(30))
        else:
            st.warning("⏳ Aguardando dados suficientes para calcular indicadores...")
        
        # Tempo de atualização rápido
        time.sleep(2)
