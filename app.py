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
    m = tf.keras.models.load_model('modelo_agressivo.keras')
    with open('scaler.pkl', 'rb') as f:
        s = pickle.load(f)
    return m, s

modelo, scaler = load_assets()

# 2. Interface
st.title("🤖 IA Trader Mobile")
ativo = st.selectbox("Par do TradingView", ["BTC-USD", "ETH-USD", "SOL-USD", "PETR4.SA", "VALE3.SA"])

if st.button('Ligar Analisador'):
    placeholder = st.empty()
    while True:
        # Busca dados e calcula indicadores
        df = yf.download(ativo, period="1d", interval="1m", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).mean() / df['Close'].diff().lt(0).rolling(14).mean()))
        df['EMA_200'] = df['Close'].ewm(span=200).mean()
        df['Dist_EMA'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
        df['Vol_ZScore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()
        df['Forca_Corpo'] = np.where((df['High']-df['Low']) > 0, abs(df['Close']-df['Open'])/(df['High']-df['Low']), 0)
        
        # Predição
        features = ['RSI', 'Dist_EMA', 'Vol_ZScore', 'Forca_Corpo']
        input_data = df[features].iloc[-1:].values
        input_scaled = scaler.transform(input_data)
        preds = modelo.predict(input_scaled, verbose=0)[0]
        
        with placeholder.container():
            st.metric("Preço", f"{df['Close'].iloc[-1]:.2f}")
            st.write(f"📈 Alta: {preds[2]*100:.1f}% | 📉 Baixa: {preds[1]*100:.1f}%")
            if preds[2] > 0.65: st.success("🚀 SINAL DE COMPRA")
            elif preds[1] > 0.65: st.error("📉 SINAL DE VENDA")
            st.line_chart(df['Close'].tail(30))
        
        time.sleep(5)