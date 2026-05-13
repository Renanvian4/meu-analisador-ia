import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import time
import os
from datetime import datetime, timedelta

# --- CONFIGURAÇÕES DE INTERFACE PARA TABLET ---
st.set_page_config(page_title="IA Quant Scanner Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        .main .block-container { max-width: 100%; padding-top: 0.5rem; }
        [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
        .stDataFrame { font-size: 0.75rem; }
        .reportview-container { background: #0e1117; }
    </style>
""", unsafe_allow_html=True)

# --- DIRETÓRIOS E ARQUIVOS ---
FOLDER_BRAIN = "cerebro_ia_dados"
FILE_LOG_ERROS = "memoria_blindada_erros.csv"

if not os.path.exists(FOLDER_BRAIN):
    os.makedirs(FOLDER_BRAIN)

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
    except Exception as e:
        st.error(f"Erro ao carregar arquivos da IA: {e}")
        return None, None

modelo, scaler = load_assets()

# --- MOTOR DE APRENDIZADO NATIVO (DEEP SCAN) ---
def executar_deep_scan(watchlist):
    timeframes = {
        "1m": "7d", "5m": "30d", "15m": "60d", "1h": "120d"
    }
    for ativo in watchlist:
        for tf_nome, periodo in timeframes.items():
            path = f"{FOLDER_BRAIN}/brain_{ativo}_{tf_nome}.csv"
            if os.path.exists(path): continue
            
            hist = yf.download(ativo, period=periodo, interval=tf_nome, progress=False)
            if hist.empty: continue
            
            # Cálculo de indicadores para estudo
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

# --- SISTEMA DE MEMÓRIA E FILTRO ---
def consultar_similaridade_erro(ativo, features, tipo, timeframe):
    path = f"{FOLDER_BRAIN}/brain_{ativo}_{timeframe}.csv"
    if not os.path.exists(path): return False
    
    # Carrega memória de falhas (resultado 0 para compra, 1 para venda)
    df = pd.read_csv(path, names=['r', 'd', 'c', 'res'])
    alvo_erro = 0 if tipo == "COMPRA" else 1
    erros = df[df['res'] == alvo_erro]
    
    if erros.empty: return False
    
    distancias = np.linalg.norm(erros[['r', 'd', 'c']].values - features[:3], axis=1)
    return np.min(distancias) < 0.05

# --- INTERFACE PRINCIPAL ---
if 'log_visual' not in st.session_state:
    st.session_state.log_visual = []
if 'audit_queue' not in st.session_state:
    st.session_state.audit_queue = []

st.title("🛰️ IA QUANT ADAPTATIVA")

with st.sidebar:
    st.header("⚙️ Scanner")
    lista_ativos = st.multiselect("Ativos", ["WIN=F", "WDO=F", "BTC-USD", "ETH-USD"], default=["WIN=F", "WDO=F"])
    tf_selecionado = st.selectbox("Timeframe Operacional", ["1m", "5m", "15m", "1h"], index=1)
    if st.button("🧠 Executar Deep Scan"):
        with st.spinner("IA aprendendo padrões históricos..."):
            executar_deep_scan(lista_ativos)
        st.success("Estudo concluído!")
    if st.button("🗑️ Limpar Log Visual"):
        st.session_state.log_visual = []
        st.rerun()

col_monitor, col_historico = st.columns([1, 1.2])

# --- LOOP DE EXECUÇÃO ---
placeholder_cards = col_monitor.empty()
placeholder_table = col_historico.empty()

if modelo is not None:
    while True:
        try:
            dados = yf.download(lista_ativos, period="2d", interval=tf_selecionado, progress=False, group_by='ticker')
            
            for ativo in lista_ativos:
                df = dados[ativo] if len(lista_ativos) > 1 else dados
                if df.empty or len(df) < 30: continue
                
                # Cálculo rápido de indicadores
                df['RSI'] = 100 - (100 / (1 + df['Close'].diff().gt(0).rolling(14).mean() / df['Close'].diff().lt(0).rolling(14).mean()))
                df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
                df['Dist'] = (df['Close'] - df['EMA200']) / df['EMA200']
                pavio = df['High'] - df['Low']
                df['Corpo'] = np.where(pavio > 0, abs(df['Close'] - df['Open']) / pavio, 0)
                
                last = df.dropna().iloc[-1:]
                f_raw = last[['RSI', 'Dist', 'Dist', 'Corpo']].values # Alinhado ao input do scaler
                f_scaled = scaler.transform(f_raw)
                
                preds = modelo.predict(f_scaled, verbose=0)[0]
                preco_at = last['Close'].values[0]
                
                tipo = "COMPRA" if preds[2] > 0.65 and preco_at > last['EMA200'].values[0] else \
                       "VENDA" if preds[1] > 0.65 and preco_at < last['EMA200'].values[0] else None
                
                if tipo and not consultar_similaridade_erro(ativo, f_raw[0], tipo, tf_selecionado):
                    info = {"Ativo": ativo, "Hora": datetime.now().strftime("%H:%M"), "Preço": f"{preco_at:.2f}", "Tipo": tipo}
                    
                    # Evita duplicados no mesmo minuto
                    if not any(l['Ativo'] == ativo and l['Hora'] == info['Hora'] for l in st.session_state.log_visual):
                        st.session_state.log_visual.insert(0, info)
                        st.session_state.audit_queue.append({
                            "ativo": ativo, "entrada": preco_at, "tipo": tipo, "f": f_raw[0],
                            "check_at": datetime.now() + timedelta(minutes=5), "done": False
                        })

            # Auditoria de Erros (Self-Learning)
            for a in st.session_state.audit_queue:
                if not a['done'] and datetime.now() >= a['check_at']:
                    v_hist = yf.download(a['ativo'], period="1d", interval="1m", progress=False).iloc[-1:]
                    ganhou = (a['tipo'] == "COMPRA" and v_hist['Close'].values[0] > a['entrada']) or \
                             (a['tipo'] == "VENDA" and v_hist['Close'].values[0] < a['entrada'])
                    if not ganhou:
                        # Salva erro na memória blindada
                        with open(f"{FOLDER_BRAIN}/brain_{a['ativo']}_{tf_selecionado}.csv", "a") as f:
                            f.write(f"{a['f'][0]},{a['f'][1]},{a['f'][2]},0\n")
                    a['done'] = True

            # Renderização
            with placeholder_cards.container():
                for s in st.session_state.log_visual[:5]:
                    cor = "#00FF00" if s['Tipo'] == "COMPRA" else "#FF4B4B"
                    st.markdown(f"<div style='border-left:4px solid {cor}; padding:10px; background:#1e1e1e; margin-bottom:5px;'><b>{s['Ativo']}</b>: {s['Tipo']} @ {s['Preço']}</div>", unsafe_allow_html=True)
            
            placeholder_table.dataframe(pd.DataFrame(st.session_state.log_visual), use_container_width=True)
            
            time.sleep(20)
        except Exception as e:
            time.sleep(10)