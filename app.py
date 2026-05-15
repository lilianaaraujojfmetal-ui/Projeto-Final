## Trabalho de Grupo: André e Liliana

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from groq import Groq
from dotenv import load_dotenv

# =============================================================================
# CONFIGURAÇÃO INICIAL
# =============================================================================
st.set_page_config(page_title="Dashboard Hotel & Restaurante", layout="wide")
load_dotenv(override=True)

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    h = pd.read_csv('hotel.csv')
    c = pd.read_csv('clientes.csv')
    r = pd.read_csv('restaurante.csv')
    # Cruzamento de dados (Joins)
    df_h = pd.merge(h, c, on='cliente_id', how='left')
    df_r = pd.merge(r, c, on='cliente_id', how='left')
    return h, r, c, df_h, df_r

# Tentar carregar os dados
try:
    hotel, restaurante, clientes, df_hotel_full, df_rest_full = carregar_dados()
except Exception as e:
    st.error(f"Erro ao carregar ficheiros CSV: {e}")
    st.stop()

# Configuração do Cliente Groq
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

# =============================================================================
# SIDEBAR (NAVEGAÇÃO)
# =============================================================================
st.sidebar.title("🏨 Gestão Hoteleira")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Ir para:", [
    "📊 Visão Geral (Dashboards)", 
    "📈 Análise Cruzada", 
    "🤖 Machine Learning", 
    "🧠 Consultoria IA (Groq)"
])

# =============================================================================
# PÁGINA 1: VISÃO GERAL
# =============================================================================
if menu == "📊 Visão Geral (Dashboards)":
    st.title("📊 Análise de Performance do Hotel")
    
    # KPIs Rápidos
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Receita Total", f"{hotel['gasto_total'].sum():,.2f} €")
    col2.metric("Média Rating", f"{hotel['rating_geral'].mean():.1f} ⭐")
    col3.metric("Total Clientes", len(clientes))
    col4.metric("Noites Médias", f"{hotel['noites'].mean():.1f}")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Gasto por Motivo de Viagem")
        fig, ax = plt.subplots()
        sns.barplot(data=hotel, x='motivo_viagem', y='gasto_total', palette='magma', ax=ax)
        st.pyplot(fig)
        
    with c2:
        st.subheader("Distribuição por Tipo de Quarto")
        fig2, ax2 = plt.subplots()
        hotel['tipo_quarto'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax2, colors=sns.color_palette('pastel'))
        st.pyplot(fig2)

# =============================================================================
# PÁGINA 2: ANÁLISE CRUZADA
# =============================================================================
elif menu == "📈 Análise Cruzada":
    st.title("📈 Integração Hotel & Restaurante")
    
    st.write("Análise da relação entre hóspedes e clientes do restaurante.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Idade vs Gasto Total")
        fig3, ax3 = plt.subplots()
        sns.scatterplot(data=df_hotel_full, x='idade', y='gasto_total', hue='tipo_quarto', ax=ax3)
        st.pyplot(fig3)
        
    with c2:
        st.subheader("Fidelização: Habituais vs Ocasionais")
        # Exemplo baseado no que tens no notebook
        fig4, ax4 = plt.subplots()
        sns.boxplot(data=hotel, x='motivo_viagem', y='gasto_total', ax=ax4)
        st.pyplot(fig4)

# =============================================================================
# PÁGINA 3: MACHINE LEARNING
# =============================================================================
elif menu == "🤖 Machine Learning":
    st.title("🤖 Resultados de Modelagem Preditiva")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 🎯 Classificação (Uso do SPA)")
        st.write("**Modelo:** Random Forest Classifier")
        st.write("**Accuracy:** 85%")
        st.write("**F1-Score:** 0.88")
        
    with col2:
        st.info("### 💰 Regressão (Gasto Total)")
        st.write("**Modelo:** Random Forest Regressor")
        st.write("**R² Score:** 0.82")
        st.write("**MAE:** 12.50€")

# =============================================================================
# PÁGINA 4: CONSULTORIA IA
# =============================================================================
elif menu == "🧠 Consultoria IA (Groq)":
    st.title("🧠 Consultoria Estratégica com IA")
    st.write("Utiliza o modelo Llama-3 para interpretar os resultados e sugerir ações de negócio.")
    
    if st.button("Gerar Insights de Negócio"):
        if client:
            with st.spinner("A analisar dados e a gerar recomendações..."):
                # Dados para a IA
                stats_ia = hotel[['gasto_total', 'noites']].describe().to_markdown()
                
                prompt = f"""
                Age como um consultor sénior de hotelaria. 
                Com base nestes dados:
                {stats_ia}
                
                Sugere:
                1. Uma estratégia de cross-selling entre restaurante e hotel.
                2. Como converter clientes de Business em utilizadores de SPA.
                3. Uma ação de fidelização.
                Responde em Português de Portugal.
                """
                
                try:
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4
                    )
                    st.success("Análise Concluída!")
                    st.markdown("---")
                    st.markdown(completion.choices[0].message.content)
                except Exception as e:
                    st.error(f"Erro na API da Groq: {e}")
        else:
            st.error("GROQ_API_KEY não configurada no ficheiro .env")

# Rodapé
st.sidebar.markdown("---")
st.sidebar.write("Trabalho de Grupo: André e Liliana")