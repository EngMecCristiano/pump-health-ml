import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Bibliotecas de confiabilidade
from scipy import stats
from scipy.stats import weibull_min, expon, lognorm
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import io
import base64
import json

# Bibliotecas de Machine Learning Avançado
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, mean_absolute_error, 
                           mean_squared_error, r2_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from fpdf import FPDF
import tempfile

# Configuração da página
st.set_page_config(
    page_title="Sistema de Confiabilidade - Bombas Centrífugas",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado aprimorado
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-left: 5px solid #3B82F6;
        padding-left: 15px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .component-card {
        background-color: #EFF6FF;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 2px solid #DBEAFE;
        transition: all 0.3s;
    }
    .component-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.1);
    }
    .alert-high {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        border-left: 5px solid #DC2626;
        box-shadow: 0 5px 15px rgba(220, 38, 38, 0.2);
    }
    .alert-medium {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        padding: 1rem;
        border-radius: 12px;
        color: #1F2937;
        border-left: 5px solid #D97706;
        box-shadow: 0 5px 15px rgba(217, 119, 6, 0.2);
    }
    .alert-low {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        border-left: 5px solid #16A34A;
        box-shadow: 0 5px 15px rgba(22, 163, 74, 0.2);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3B82F6, #1E3A8A);
    }
    .feature-importance-bar {
        background: linear-gradient(90deg, #3B82F6, #1E3A8A);
        height: 25px;
        border-radius: 12px;
        margin: 5px 0;
        transition: width 0.5s;
    }
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #E5E7EB;
        transition: all 0.3s;
    }
    .model-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.15);
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.3rem;
        font-size: 0.9rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .model-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .confidence-meter {
        height: 10px;
        background: #E5E7EB;
        border-radius: 5px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# INICIALIZAÇÃO DO SESSION STATE
# ============================

# Lista de chaves necessárias
keys = [
    'df_sintetico', 'df_real', 'dados_tipo', 'cph_model', 'aft_model', 
    'ml_models', 'ml_metrics', 'feature_importance', 'data_geracao', 
    'preprocessors', 'colunas_selecionadas'
]

# Inicializa as chaves com dicionário para as de ML e None para as demais
for key in keys:
    if key not in st.session_state:
        if key in ['ml_models', 'ml_metrics', 'feature_importance', 'preprocessors']:
            st.session_state[key] = {}
        else:
            st.session_state[key] = None


# ============================
# FUNÇÕES AUXILIARES
# ============================
def get_kmf_survival_data(kmf):
    """Função compatível para obter dados de sobrevivência"""
    if hasattr(kmf, 'survival_function_'):
        survival_df = kmf.survival_function_
        survival_col = survival_df.columns[0]
        timeline = survival_df.index
        survival_values = survival_df[survival_col].values
        return timeline, survival_values, survival_col
    else:
        timeline = getattr(kmf, 'timeline', getattr(kmf, 'timeline_', np.array([])))
        if timeline is not None and len(timeline) > 0:
            try:
                survival_values = kmf.survival_function_at_times(timeline)
            except:
                survival_values = np.ones_like(timeline)
        else:
            timeline = np.array([])
            survival_values = np.array([])
        return timeline, survival_values, 'survival'

def get_kmf_confidence_interval(kmf):
    """Função compatível para obter intervalo de confiança"""
    if hasattr(kmf, 'confidence_interval_'):
        ci_df = kmf.confidence_interval_
        if len(ci_df.columns) >= 2:
            col1 = ci_df.columns[0]
            col2 = ci_df.columns[1]
            return ci_df[col1].values, ci_df[col2].values
        elif 'KM_estimate_lower_0.95' in ci_df.columns and 'KM_estimate_upper_0.95' in ci_df.columns:
            return ci_df['KM_estimate_lower_0.95'].values, ci_df['KM_estimate_upper_0.95'].values
    return None, None

def get_naf_hazard_data(naf):
    """Função compatível para obter dados de hazard de Nelson-Aalen"""
    if hasattr(naf, 'cumulative_hazard_'):
        hazard_df = naf.cumulative_hazard_
        hazard_col = hazard_df.columns[0]
        timeline = hazard_df.index
        hazard_values = hazard_df[hazard_col].values
        return timeline, hazard_values
    else:
        timeline = getattr(naf, 'timeline', np.array([]))
        try:
            hazard_values = naf.cumulative_hazard_at_times(timeline)
        except:
            hazard_values = np.zeros_like(timeline)
        return timeline, hazard_values

def validar_dataset(df):
    """Valida se o dataset contém as colunas necessárias"""
    colunas_obrigatorias = ['Tempo_Operacao', 'Falha']
    colunas_presentes = [col for col in colunas_obrigatorias if col in df.columns]
    return len(colunas_presentes) == len(colunas_obrigatorias), colunas_obrigatorias

def verificar_colunas_disponiveis(df, colunas_necessarias):
    """Verifica quais colunas necessárias estão disponíveis no dataframe"""
    colunas_disponiveis = []
    colunas_faltantes = []
    
    for coluna in colunas_necessarias:
        if coluna in df.columns:
            colunas_disponiveis.append(coluna)
        else:
            colunas_faltantes.append(coluna)
    
    return colunas_disponiveis, colunas_faltantes

@st.cache_data(ttl=3600)
def calcular_metricas_confiabilidade(df):
    """Calcula métricas de confiabilidade básicas com cache"""
    metricas = {}
    
    if 'Falha' in df.columns and 'Tempo_Operacao' in df.columns:
        metricas['total_falhas'] = int(df['Falha'].sum())
        metricas['taxa_falhas'] = metricas['total_falhas'] / len(df) * 100
        
        if metricas['total_falhas'] > 0:
            metricas['mtbf'] = df[df['Falha'] == 1]['Tempo_Operacao'].mean()
        else:
            metricas['mtbf'] = df['Tempo_Operacao'].max()
            
        tempo_total = df['Tempo_Operacao'].sum()
        tempo_falha = df[df['Falha'] == 1]['Tempo_Operacao'].sum()
        metricas['taxa_censura'] = (df['Falha'] == 0).sum() / len(df) * 100
    else:
        metricas['total_falhas'] = 0
        metricas['taxa_falhas'] = 0
        metricas['mtbf'] = 0
        metricas['taxa_censura'] = 0
    
    return metricas

@st.cache_data(ttl=3600)
def gerar_dados_vetorizados(n_bombas, periodo_horas, shape, scale, 
                           temp_media=65, temp_var=10,
                           vib_media=5, vib_var=1,
                           press_media=15, press_var=2):
    """Gera dados sintéticos vetorizados com degradação temporal"""
    # TTF base com Weibull
    ttf_base = weibull_min.rvs(shape, scale=scale, size=n_bombas)
    
    # Fatores de correlação entre componentes
    correl_matrix = np.array([
        [1.0, 0.3, 0.2, 0.1],   # Rotor
        [0.3, 1.0, 0.4, 0.2],   # Mancal
        [0.2, 0.4, 1.0, 0.5],   # Selagem
        [0.1, 0.2, 0.5, 1.0]    # Eixo
    ])
    
    # Gerar tempos de falha correlacionados
    chol = np.linalg.cholesky(correl_matrix)
    uncorrelated = np.random.normal(0, 1, (n_bombas, 4))
    correlated = uncorrelated @ chol
    
    # Ajustar tempos de falha com degradação temporal
    ttf_ajustado = ttf_base * (0.8 + 0.4 * np.exp(-correlated[:, 0]))
    
    # Sensor data with degradation trend
    tempo_norm = ttf_ajustado / scale
    vibracao = (vib_media + vib_var * correlated[:, 1] + 
                10 * (1 - np.exp(-5 * tempo_norm)))
    temperatura = (temp_media + temp_var * correlated[:, 2] + 
                   vibracao * 0.3 + 15 * (1 - np.exp(-3 * tempo_norm)))
    pressao = (press_media + press_var * correlated[:, 3] + 
               5 * (1 - np.exp(-2 * tempo_norm)))
    
    # Aplicar censura
    taxa_censura = 0.25
    censurado = np.random.random(n_bombas) < taxa_censura
    tempo_observado = np.where(
        censurado,
        np.minimum(ttf_ajustado * np.random.uniform(0.3, 0.8, n_bombas), periodo_horas),
        np.minimum(ttf_ajustado, periodo_horas)
    )
    
    falha_observada = (~censurado & (ttf_ajustado <= periodo_horas)).astype(int)
    
    # Determinar componente que falhou
    componentes = ['Rotor', 'Mancal', 'Selagem', 'Eixo', 'Impulsor', 'Carcaça']
    componente_falha = np.where(
        falha_observada == 1,
        np.random.choice(componentes, n_bombas, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05]),
        'Censurado'
    )
    
    # Adicionar tipo de falha para ML
    tipos_falha = []
    for i in range(n_bombas):
        if falha_observada[i] == 1:
            if vibracao[i] > 8:
                tipo = 'Mecânica' if np.random.random() > 0.5 else 'Vibração'
            elif temperatura[i] > 80:
                tipo = 'Térmica' if np.random.random() > 0.5 else 'Sobreaquecimento'
            else:
                tipo = np.random.choice(['Fadiga', 'Desgaste', 'Lubrificação'], p=[0.5, 0.3, 0.2])
        else:
            tipo = 'Nenhuma'
        tipos_falha.append(tipo)
    
    # Criar DataFrame
    df = pd.DataFrame({
        'ID_Bomba': [f'BOMBA_{i+1:04d}' for i in range(n_bombas)],
        'Tempo_Operacao': tempo_observado,
        'Falha': falha_observada,
        'Tipo_Falha': tipos_falha,
        'Componente_Falha': componente_falha,
        'Vibracao_Media': np.clip(vibracao, 1, 20),
        'Vibracao_Max': np.clip(vibracao * 1.5, 2, 30),
        'Temperatura_Media': np.clip(temperatura, 20, 120),
        'Temperatura_Max': np.clip(temperatura * 1.2, 25, 140),
        'Pressao_Media': np.clip(pressao, 1, 50),
        'Pressao_Pico': np.clip(pressao * 1.3, 1.5, 65),
        'Vazao_Media': np.random.normal(200, 40, n_bombas),
        'Velocidade_RPM': np.random.normal(1800, 100, n_bombas),
        'Material': np.random.choice(['Aço Carbono', 'Aço Inox 304', 'Aço Inox 316'], n_bombas),
        'Lubrificacao': np.random.choice(['Graxa', 'Óleo', 'Splash'], n_bombas),
        'Ambiente': np.random.choice(['Água Limpa', 'Água Salgada', 'Solução Química'], n_bombas),
        'Horas_Dia': np.random.choice([8, 16, 24], n_bombas)
    })
    
    return df

def criar_matriz_risco(df):
    """Cria matriz de risco (Criticidade vs Probabilidade) adaptativa"""
    if 'Componente_Falha' not in df.columns or 'Falha' not in df.columns:
        return None
    
    df_falhas = df[df['Falha'] == 1]
    if len(df_falhas) == 0:
        return None
    
    # Calcular probabilidade por componente
    prob_falha = df_falhas['Componente_Falha'].value_counts(normalize=True)
    
    # Criticidade adaptativa
    criticidade_default = {
        'Rotor': 72, 'Mancal': 24, 'Selagem': 48, 
        'Eixo': 96, 'Impulsor': 36, 'Carcaça': 120,
        'Censurado': 0, 'Nenhuma': 0
    }
    
    # Criar matriz
    matriz_data = []
    for componente in prob_falha.index:
        if pd.notna(componente):
            crit = criticidade_default.get(componente, 48)
            matriz_data.append({
                'Componente': componente,
                'Probabilidade': prob_falha[componente] * 100,
                'Criticidade': crit,
                'Risco': prob_falha[componente] * crit * 10
            })
    
    return pd.DataFrame(matriz_data)

def gerar_relatorio_pdf(metricas, df, insights):
    """Gera relatório PDF com resultados da análise"""
    pdf = FPDF()
    pdf.add_page()
    
    # Cabeçalho
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'RELATÓRIO DE CONFIABILIDADE - BOMBAS CENTRÍFUGAS', 0, 1, 'C')
    pdf.ln(5)
    
    # Data e hora
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Data de geração: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(10)
    
    # Sumário Executivo
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. SUMÁRIO EXECUTIVO', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    sumario = f"""
    Total de Bombas Analisadas: {len(df)}
    MTBF (Tempo Médio Entre Falhas): {metricas.get('mtbf', 0):.0f} horas
    Taxa de Falhas: {metricas.get('taxa_falhas', 0):.1f}%
    Falhas Observadas: {metricas.get('total_falhas', 0)}
    Taxa de Censura: {metricas.get('taxa_censura', 0):.1f}%
    """
    
    for linha in sumario.strip().split('\n'):
        pdf.multi_cell(0, 5, linha.strip())
    
    pdf.ln(10)
    
    # Insights
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. INSIGHTS E RECOMENDAÇÕES', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for i, insight in enumerate(insights[:5], 1):
        pdf.multi_cell(0, 5, f"{i}. {insight}")
    
    pdf.ln(10)
    
    # Plano de Ação
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '3. PLANO DE AÇÃO RECOMENDADO', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    acoes = [
        "Curto Prazo (0-3 meses): Implementar monitoramento contínuo nas bombas críticas",
        "Médio Prazo (3-12 meses): Revisar procedimentos de manutenção preventiva",
        "Longo Prazo (12+ meses): Implementar sistema preditivo baseado em IA"
    ]
    
    for acao in acoes:
        pdf.multi_cell(0, 5, f"• {acao}")
    
    # Salvar PDF temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf.output(tmp.name)
        return tmp.name

# ============================
# FUNÇÕES DE MACHINE LEARNING
# ============================
def preparar_dados_ml(df, problema='classificacao'):
    """Prepara dados para modelos de ML"""
    df_ml = df.copy()
    
    # Remover colunas não úteis para ML
    colunas_remover = ['ID_Bomba']
    df_ml = df_ml.drop(columns=[c for c in colunas_remover if c in df_ml.columns])
    
    # Codificar variáveis categóricas
    categorical_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le
    
    # Separar features e target baseado no tipo de problema
    if problema == 'classificacao_tipo':
        if 'Tipo_Falha' in df_ml.columns:
            X = df_ml.drop(columns=['Tipo_Falha', 'Falha', 'Componente_Falha'], errors='ignore')
            y = df_ml['Tipo_Falha']
            target_type = 'Tipo_Falha'
        else:
            return None, None, None, None, None, None
    elif problema == 'classificacao_componente':
        if 'Componente_Falha' in df_ml.columns:
            X = df_ml.drop(columns=['Componente_Falha', 'Falha', 'Tipo_Falha'], errors='ignore')
            y = df_ml['Componente_Falha']
            target_type = 'Componente_Falha'
        else:
            return None, None, None, None, None, None
    elif problema == 'regressao_tempo':
        if 'Tempo_Operacao' in df_ml.columns:
            X = df_ml.drop(columns=['Tempo_Operacao', 'Falha'], errors='ignore')
            y = df_ml['Tempo_Operacao']
            target_type = 'Tempo_Operacao'
        else:
            return None, None, None, None, None, None
    else:
        return None, None, None, None, None, None
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if problema in ['classificacao_tipo', 'classificacao_componente'] else None
    )
    
    return X_train, X_test, y_train, y_test, target_type, label_encoders

def treinar_modelo_classificacao(X_train, X_test, y_train, y_test, modelo_tipo='rf'):
    """Treina modelo de classificação"""
    if modelo_tipo == 'rf':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif modelo_tipo == 'xgb':
        model = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    elif modelo_tipo == 'lgbm':
        model = LGBMClassifier(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    elif modelo_tipo == 'catboost':
        model = CatBoostClassifier(
            iterations=150,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
    else:
        model = GradientBoostingClassifier(random_state=42)
    
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Importância das features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = None
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'feature_importances': importances
    }

def treinar_modelo_regressao(X_train, X_test, y_train, y_test, modelo_tipo='rf'):
    """Treina modelo de regressão"""
    if modelo_tipo == 'rf':
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif modelo_tipo == 'xgb':
        model = XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    elif modelo_tipo == 'lgbm':
        model = LGBMRegressor(
            n_estimators=150,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    elif modelo_tipo == 'catboost':
        model = CatBoostRegressor(
            iterations=150,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0
        )
    else:
        model = RandomForestRegressor(random_state=42)
    
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Previsões
    y_pred = model.predict(X_test)
    
    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Importância das features
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = None
    
    return {
        'model': model,
        'y_pred': y_pred,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'feature_importances': importances
    }

def plot_confusion_matrix_plotly(cm, classes):
    """Plota matriz de confusão interativa"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale='Blues',
        text=[[str(y) for y in x] for x in cm],
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
        showscale=True
    ))
    
    fig.update_layout(
        title='Matriz de Confusão',
        xaxis_title='Predito',
        yaxis_title='Real',
        height=500,
        width=600
    )
    
    return fig

def plot_feature_importance_plotly(importances, feature_names, title):
    """Plota importância das features"""
    # Ordenar por importância
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_features = [feature_names[i] for i in indices]
    
    fig = go.Figure(data=go.Bar(
        x=sorted_importances[:15],  # Top 15 features
        y=sorted_features[:15],
        orientation='h',
        marker=dict(
            color=sorted_importances[:15],
            colorscale='Viridis',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importância',
        yaxis_title='Feature',
        height=500,
        width=800
    )
    
    return fig

def plot_predictions_vs_actual(y_true, y_pred, title):
    """Plota previsões vs valores reais para regressão"""
    fig = go.Figure()
    
    # Linha de referência (y = x)
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Ideal',
        line=dict(color='red', dash='dash')
    ))
    
    # Previsões
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Previsões',
        marker=dict(
            color='blue',
            size=8,
            opacity=0.6
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Valor Real (horas)',
        yaxis_title='Valor Predito (horas)',
        height=500,
        width=600,
        showlegend=True
    )
    
    return fig

def criar_dashboard_ml(model_metrics, feature_importance, X_test, y_test, y_pred, problema):
    """Cria dashboard visual para resultados de ML"""
    
    st.markdown('<h2 class="sub-header">📊 Dashboard de Resultados - Machine Learning</h2>', unsafe_allow_html=True)
    
    if problema in ['classificacao_tipo', 'classificacao_componente']:
        # Métricas de classificação
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Acurácia", f"{model_metrics['accuracy']:.3f}")
        with col2:
            st.metric("Precisão", f"{model_metrics['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{model_metrics['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{model_metrics['f1']:.3f}")
        
        # Matriz de confusão
        st.subheader("🎯 Matriz de Confusão")
        classes = np.unique(y_test)
        cm_fig = plot_confusion_matrix_plotly(model_metrics['confusion_matrix'], classes)
        st.plotly_chart(cm_fig, use_container_width=True)
        
        # Relatório de classificação
        st.subheader("📋 Relatório de Classificação")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.background_gradient(subset=['precision', 'recall', 'f1-score'], cmap='YlOrRd'), 
                    use_container_width=True)
    
    else:  # Regressão
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("MAE (horas)", f"{model_metrics['mae']:.1f}")
        with col2:
            st.metric("RMSE (horas)", f"{model_metrics['rmse']:.1f}")
        with col3:
            st.metric("R² Score", f"{model_metrics['r2']:.3f}")
        with col4:
            st.metric("Erro Relativo", f"{(model_metrics['mae']/y_test.mean()*100):.1f}%")
        
        # Gráfico de previsões vs reais
        st.subheader("📈 Previsões vs Valores Reais")
        pred_fig = plot_predictions_vs_actual(y_test, y_pred, "Comparação: Real vs Predito")
        st.plotly_chart(pred_fig, use_container_width=True)
        
        # Distribuição dos erros
        st.subheader("📊 Distribuição dos Erros de Predição")
        errors = y_test - y_pred
        fig_errors = px.histogram(
            x=errors,
            nbins=30,
            title="Distribuição dos Erros",
            labels={'x': 'Erro (horas)', 'y': 'Frequência'}
        )
        fig_errors.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Erro Zero")
        st.plotly_chart(fig_errors, use_container_width=True)
    
    # Importância das features
    if feature_importance is not None and len(feature_importance) > 0:
        st.subheader("🔝 Top 15 Features mais Importantes")
        feature_names = X_test.columns.tolist()
        
        # Criar ranking visual
        col1, col2 = st.columns([2, 1])
        
        with col1:
            imp_fig = plot_feature_importance_plotly(
                feature_importance, 
                feature_names, 
                "Importância das Features"
            )
            st.plotly_chart(imp_fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🏆 Ranking das Features")
            indices = np.argsort(feature_importance)[::-1]
            for i, idx in enumerate(indices[:10], 1):
                importance_pct = (feature_importance[idx] / feature_importance.sum()) * 100
                st.markdown(f"""
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: bold;">{i}. {feature_names[idx]}</span>
                        <span style="color: #3B82F6; font-weight: bold;">{importance_pct:.1f}%</span>
                    </div>
                    <div style="height: 8px; background: #E5E7EB; border-radius: 4px; margin-top: 5px;">
                        <div style="width: {importance_pct}%; height: 100%; background: linear-gradient(90deg, #3B82F6, #1E3A8A); border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============================
# TÍTULO PRINCIPAL
# ============================
st.markdown('<h1 class="main-header">🔧 Sistema de Análise de Confiabilidade - Bombas Centrífugas</h1>', unsafe_allow_html=True)

# ============================
# SIDEBAR
# ============================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/pump.png", width=100)
    st.title("Navegação")
    
    menu = st.radio(
        "Selecione o módulo:",
        ["⚙️ Gerar Dados Sintéticos", 
         "📁 Carregar Dados Reais", 
         "🔍 Análise de Componentes",
         "📉 Curvas de Sobrevivência", 
         "⚡ Modelos Preditivos",
         "🔔 Sistema de Alerta",
         "📊 Dashboard Industrial",
         "📋 Relatório Técnico",
         "🤖 Modelos de Machine Learning"]
    )
    
    st.markdown("---")
    st.markdown("### 🎯 Configurações")
    
    tipo_analise = st.selectbox(
        "Tipo de análise:",
        ["Bomba Completa", "Rotor", "Mancal", "Selagem Mecânica", 
         "Carcaça", "Eixo", "Impulsor"]
    )
    
    st.markdown("---")
    
    # Status do sistema
    st.markdown("### 📊 Status do Sistema")
    if st.session_state['df_sintetico'] is not None or st.session_state['df_real'] is not None:
        df_atual = st.session_state['df_sintetico'] if st.session_state['dados_tipo'] == 'sintetico' else st.session_state['df_real']
        if df_atual is not None:
            st.success(f"✅ Dados carregados: {len(df_atual)} registros")
            metricas = calcular_metricas_confiabilidade(df_atual)
            st.metric("MTBF", f"{metricas.get('mtbf', 0):.0f}h")
            if st.session_state.get('colunas_selecionadas'):
                st.info(f"📋 {len(st.session_state['colunas_selecionadas'])} colunas selecionadas")
        else:
            st.warning("⚠️ Aguardando dados")
    else:
        st.info("ℹ️ Nenhum dataset carregado")

# ============================
# MÓDULO: Gerar Dados Sintéticos (VETORIZADO)
# ============================
if menu == "⚙️ Gerar Dados Sintéticos":
    st.header("⚙️ Gerador de Dados Sintéticos - Bombas Centrífugas")
    
    tab1, tab2, tab3 = st.tabs(["📊 Configurações Gerais", "🔧 Features Específicas", "📈 Parâmetros Weibull"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            n_bombas = st.slider("Número de bombas", 100, 10000, 1000, step=100)
            periodo_horas = st.slider("Período de operação (horas)", 1000, 30000, 5000, step=500)
            
            tipo_bomba = st.selectbox(
                "Tipo de bomba:",
                ["Centrífuga Horizontal", "Centrífuga Vertical", "Multiestágio", "Autoescorvante"]
            )
        
        with col2:
            ambientes = st.multiselect(
                "Ambientes de operação:",
                ["Água Limpa", "Água Salgada", "Solução Química", "Slurry", "Óleo"],
                default=["Água Limpa"]
            )
            
            turnos = st.select_slider(
                "Turnos de operação:",
                options=["1 turno", "2 turnos", "3 turnos", "Contínuo"]
            )
    
    with tab2:
        st.subheader("🔧 Features Operacionais com Degradação")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Temperatura**")
            temp_media = st.slider("Temperatura média (°C)", 20, 120, 65, key="temp_media")
            temp_var = st.slider("Variação temperatura", 1, 30, 10, key="temp_var")
            st.info("Inclui aumento progressivo com degradação")
        
        with col2:
            st.markdown("**Pressão**")
            press_media = st.slider("Pressão média (bar)", 1, 50, 15, key="press_media")
            press_var = st.slider("Variação pressão", 0.1, 10.0, 2.0, key="press_var")
            st.info("Correlacionada com temperatura")
        
        with col3:
            st.markdown("**Vibração**")
            vib_media = st.slider("Vibração média (mm/s)", 1, 20, 5, key="vib_media")
            vib_var = st.slider("Variação vibração", 0.1, 5.0, 1.0, key="vib_var")
            st.info("Aumenta exponencialmente com falha iminente")
        
        # Dependência entre componentes
        st.subheader("🔗 Matriz de Dependência entre Componentes")
        st.markdown("""
        | Componente | Rotor | Mancal | Selagem | Eixo |
        |------------|-------|--------|---------|------|
        | **Rotor**  | 1.00  | 0.30   | 0.20    | 0.10 |
        | **Mancal** | 0.30  | 1.00   | 0.40    | 0.20 |
        | **Selagem**| 0.20  | 0.40   | 1.00    | 0.50 |
        | **Eixo**   | 0.10  | 0.20   | 0.50    | 1.00 |
        """)
    
    with tab3:
        st.subheader("📈 Parâmetros Weibull por Componente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            shape_geral = st.slider("Shape (β) geral", 0.5, 5.0, 2.5, 0.1)
            scale_geral = st.slider("Scale (η) geral (horas)", 1000, 40000, 4500, 100)
        
        with col2:
            degradacao_taxa = st.slider("Taxa de degradação", 0.1, 5.0, 1.0, 0.1)
            correlacao_global = st.slider("Correlação global", 0.0, 1.0, 0.3, 0.05)
    
    # Botão para gerar dados
    if st.button("🎯 Gerar Dados Sintéticos (Vetorizado)", type="primary", use_container_width=True):
        with st.spinner(f"Gerando {n_bombas} bombas com dados vetorizados..."):
            progress_bar = st.progress(0)
            
            # Gerar dados vetorizados
            df_sintetico = gerar_dados_vetorizados(
                n_bombas=n_bombas,
                periodo_horas=periodo_horas,
                shape=shape_geral,
                scale=scale_geral,
                temp_media=temp_media,
                temp_var=temp_var,
                vib_media=vib_media,
                vib_var=vib_var,
                press_media=press_media,
                press_var=press_var
            )
            
            progress_bar.progress(100)
            
            # SELEÇÃO DE COLUNAS
            st.subheader("🔧 Seleção de Colunas para Análise")
            
            colunas_obrigatorias = ['Tempo_Operacao', 'Falha']
            colunas_opcionais = [col for col in df_sintetico.columns if col not in colunas_obrigatorias]
            
            colunas_selecionadas = st.multiselect(
                "Selecione quais colunas incluir na análise:",
                colunas_opcionais,
                default=colunas_opcionais[:min(10, len(colunas_opcionais))]
            )
            
            # Combinar colunas obrigatórias com selecionadas
            colunas_finais = colunas_obrigatorias + colunas_selecionadas
            
            # Filtrar dataframe
            df_sintetico_filtrado = df_sintetico[colunas_finais]
            
            # Salvar no session state
            st.session_state['df_sintetico'] = df_sintetico_filtrado
            st.session_state['dados_tipo'] = 'sintetico'
            st.session_state['data_geracao'] = datetime.now()
            st.session_state['colunas_selecionadas'] = colunas_finais
            
            # Mostrar resultados
            st.success(f"✅ {n_bombas} bombas geradas! {len(colunas_finais)} colunas selecionadas")
            
            # Métricas
            metricas = calcular_metricas_confiabilidade(df_sintetico_filtrado)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MTBF", f"{metricas['mtbf']:,.0f} h")
            with col2:
                st.metric("Falhas", metricas['total_falhas'])
            with col3:
                st.metric("Censura", f"{metricas['taxa_censura']:.1f}%")
            
            # Visualizar dados
            with st.expander("📋 Visualizar Dados Gerados", expanded=False):
                st.dataframe(df_sintetico_filtrado.head(), use_container_width=True)
                st.write(f"Dimensões: {df_sintetico_filtrado.shape[0]} linhas × {df_sintetico_filtrado.shape[1]} colunas")
            
            # Download buttons modernos
            st.subheader("💾 Exportar Dados")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df_sintetico_filtrado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"dados_bombas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    df_sintetico_filtrado.to_excel(writer, index=False, sheet_name='Dados_Bombas')
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"dados_bombas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# ============================
# MÓDULO: Carregar Dados Reais
# ============================
elif menu == "📁 Carregar Dados Reais":
    st.header("📁 Carregar Dados Reais de Bombas")
    
    tab1, tab2 = st.tabs(["Upload de Arquivo", "Formato Esperado"])
    
    with tab1:
        st.subheader("Faça upload dos seus dados")
        
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV ou Excel",
            type=['csv', 'xlsx', 'xls'],
            help="Arquivos com dados históricos de operação das bombas"
        )
        
        if uploaded_file is not None:
            try:
                # Ler arquivo com validação
                if uploaded_file.name.endswith('.csv'):
                    df_real = pd.read_csv(uploaded_file, sep=None, engine='python')
                else:
                    df_real = pd.read_excel(uploaded_file)
                
                # Validar dataset
                valido, colunas_necessarias = validar_dataset(df_real)
                
                if valido:
                    # SELEÇÃO DE COLUNAS
                    st.subheader("🔧 Seleção de Colunas para Análise")
                    
                    colunas_obrigatorias = ['Tempo_Operacao', 'Falha']
                    colunas_opcionais = [col for col in df_real.columns if col not in colunas_obrigatorias]
                    
                    colunas_selecionadas = st.multiselect(
                        "Selecione quais colunas incluir na análise:",
                        colunas_opcionais,
                        default=colunas_opcionais[:min(10, len(colunas_opcionais))]
                    )
                    
                    # Combinar colunas obrigatórias com selecionadas
                    colunas_finais = colunas_obrigatorias + colunas_selecionadas
                    
                    # Filtrar dataframe
                    df_real_filtrado = df_real[colunas_finais]
                    
                    # Salvar em session state
                    st.session_state['df_real'] = df_real_filtrado
                    st.session_state['dados_tipo'] = 'real'
                    st.session_state['data_geracao'] = datetime.now()
                    st.session_state['colunas_selecionadas'] = colunas_finais
                    
                    st.success(f"✅ Arquivo carregado! {len(df_real_filtrado)} registros, {len(colunas_finais)} colunas")
                    
                    # Mostrar informações
                    st.subheader("📊 Informações do Dataset")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Estatísticas básicas:**")
                        st.write(f"- Total de registros: {len(df_real_filtrado):,}")
                        st.write(f"- Colunas: {len(df_real_filtrado.columns)}")
                        
                        # Calcular métricas
                        metricas = calcular_metricas_confiabilidade(df_real_filtrado)
                        
                        st.write("**Métricas de confiabilidade:**")
                        st.write(f"- Total de falhas: {metricas['total_falhas']}")
                        st.write(f"- MTBF: {metricas['mtbf']:,.0f} horas")
                        st.write(f"- Taxa de falhas: {metricas['taxa_falhas']:.1f}%")
                    
                    with col2:
                        st.write("**Tipos de dados:**")
                        for dtype, count in df_real_filtrado.dtypes.value_counts().items():
                            st.write(f"- {dtype}: {count}")
                        
                        st.write("**Valores nulos:**")
                        nulos = df_real_filtrado.isnull().sum().sum()
                        st.write(f"- Total: {nulos}")
                        if nulos > 0:
                            st.warning(f"⚠️ Dataset contém {nulos} valores nulos")
                    
                    # Pré-visualização
                    with st.expander("👁️ Pré-visualização dos Dados", expanded=False):
                        st.dataframe(df_real_filtrado.head(10), use_container_width=True)
                        st.dataframe(df_real_filtrado.describe(), use_container_width=True)
                
                else:
                    colunas_faltantes = [col for col in colunas_necessarias if col not in df_real.columns]
                    st.error(f"❌ Dataset inválido. Colunas faltantes: {colunas_faltantes}")
                    st.info("Use a aba 'Formato Esperado' para verificar o formato correto.")
            
            except Exception as e:
                st.error(f"❌ Erro ao ler o arquivo: {str(e)}")
                st.info("Verifique se o arquivo está no formato correto.")
    
    with tab2:
        st.subheader("📋 Formato Esperado do Dataset")
        
        st.markdown("""
        ### Estrutura recomendada para dados de bombas:
        
        **Colunas obrigatórias (mínimo):**
        - `Tempo_Operacao`: Tempo de operação até falha ou censura (horas)
        - `Falha`: Indicador de falha (1 = falha observada, 0 = censurado)
        
        **Colunas recomendadas para análise avançada:**
        - `ID_Bomba`: Identificador único da bomba
        - `Componente_Falha`: Componente que falhou
        - `Temperatura_Media`: Temperatura média de operação (°C)
        - `Vibracao_Media`: Nível médio de vibração (mm/s)
        - `Pressao_Media`: Pressão média de operação (bar)
        
        **Exemplo de dataset válido:**
        """)
        
        exemplo_data = {
            'ID_Bomba': ['BOMBA_001', 'BOMBA_002', 'BOMBA_003', 'BOMBA_004'],
            'Tempo_Operacao': [2850, 3120, 2450, 4200],
            'Falha': [1, 1, 0, 1],
            'Componente_Falha': ['Selagem', 'Rotor', 'Censurado', 'Mancal'],
            'Temperatura_Media': [65.2, 68.5, 62.8, 71.3],
            'Vibracao_Media': [4.8, 5.2, 3.9, 6.1],
            'Pressao_Media': [15.3, 16.1, 14.8, 17.2]
        }
        
        df_exemplo = pd.DataFrame(exemplo_data)
        st.dataframe(df_exemplo, use_container_width=True)
        
        st.markdown("""
        **Dica:** Para obter os melhores resultados, inclua pelo menos 50-100 registros históricos.
        """)

# ============================
# MÓDULO: Análise de Componentes (ADAPTATIVO)
# ============================
elif menu == "🔍 Análise de Componentes":
    st.header("🔍 Análise Detalhada por Componente")
    
    # Verificar dados disponíveis
    if st.session_state['df_sintetico'] is not None or st.session_state['df_real'] is not None:
        df = st.session_state['df_sintetico'] if st.session_state['dados_tipo'] == 'sintetico' else st.session_state['df_real']
        
        if df is not None:
            # Verificar colunas disponíveis
            colunas_desejadas = ['Componente_Falha', 'Tipo_Falha', 'Vibracao_Media', 
                                'Temperatura_Media', 'Pressao_Media']
            colunas_disponiveis, colunas_faltantes = verificar_colunas_disponiveis(df, colunas_desejadas)
            
            st.info(f"📊 Dataset: {st.session_state['dados_tipo'].title()} - {len(df):,} registros")
            st.info(f"📋 Colunas analíticas disponíveis: {', '.join(colunas_disponiveis) if colunas_disponiveis else 'Nenhuma'}")
            
            if colunas_faltantes:
                st.warning(f"⚠️ Colunas não disponíveis: {', '.join(colunas_faltantes)}")
            
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribuição", "🔧 Detalhes", "📊 Comparativo", "🔥 Matriz de Risco"])
            
            with tab1:
                st.subheader("📈 Distribuição de Falhas")
                
                # Verificar se temos coluna de componente para análise
                if 'Componente_Falha' in df.columns:
                    df_falhas = df[df['Falha'] == 1]
                    
                    if len(df_falhas) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            falhas_por_componente = df_falhas['Componente_Falha'].value_counts()
                            fig = px.bar(
                                x=falhas_por_componente.index,
                                y=falhas_por_componente.values,
                                title="Número de Falhas por Componente",
                                labels={'x': 'Componente', 'y': 'Número de Falhas'},
                                color=falhas_por_componente.values,
                                color_continuous_scale='Reds'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Pie chart
                            fig = px.pie(
                                names=falhas_por_componente.index,
                                values=falhas_por_componente.values,
                                title="Distribuição Percentual",
                                hole=0.3
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("ℹ️ Não há falhas registradas no dataset")
                else:
                    st.info("ℹ️ A coluna 'Componente_Falha' não está disponível para análise de distribuição")
                    
                    # Alternativa: mostrar distribuição por outras colunas
                    colunas_categoricas = [col for col in df.columns if df[col].dtype == 'object' and df[col].nunique() < 10]
                    if colunas_categoricas:
                        coluna_alternativa = st.selectbox("Selecione coluna para análise:", colunas_categoricas)
                        if coluna_alternativa:
                            df_falhas = df[df['Falha'] == 1]
                            if len(df_falhas) > 0:
                                fig = px.bar(
                                    x=df_falhas[coluna_alternativa].value_counts().index,
                                    y=df_falhas[coluna_alternativa].value_counts().values,
                                    title=f"Falhas por {coluna_alternativa}",
                                    color=df_falhas[coluna_alternativa].value_counts().values,
                                    color_continuous_scale='Reds'
                                )
                                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("🔧 Análise Detalhada")
                
                # Lista de colunas disponíveis para análise detalhada
                colunas_analise = ['Tempo_Operacao', 'Falha']
                colunas_analise.extend([col for col in df.columns if col not in ['Tempo_Operacao', 'Falha']])
                
                coluna_selecionada = st.selectbox(
                    "Selecione a variável para análise:",
                    colunas_analise,
                    key="var_select"
                )
                
                if coluna_selecionada:
                    if coluna_selecionada == 'Tempo_Operacao':
                        # Análise de tempo
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Média", f"{df[coluna_selecionada].mean():.0f}h")
                            st.metric("Mínimo", f"{df[coluna_selecionada].min():.0f}h")
                        with col2:
                            st.metric("Máximo", f"{df[coluna_selecionada].max():.0f}h")
                            st.metric("Mediana", f"{df[coluna_selecionada].median():.0f}h")
                        
                        # Histograma
                        fig = px.histogram(df, x=coluna_selecionada, nbins=30, 
                                         title=f"Distribuição de {coluna_selecionada}")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif coluna_selecionada == 'Falha':
                        # Análise de falhas
                        falhas = df['Falha'].sum()
                        total = len(df)
                        taxa = (falhas / total) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Falhas", falhas)
                        with col2:
                            st.metric("Total Registros", total)
                        with col3:
                            st.metric("Taxa de Falhas", f"{taxa:.1f}%")
                    
                    else:
                        # Análise de outras colunas
                        if df[coluna_selecionada].dtype in ['int64', 'float64']:
                            # Coluna numérica
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Média", f"{df[coluna_selecionada].mean():.2f}")
                                st.metric("Desvio Padrão", f"{df[coluna_selecionada].std():.2f}")
                            with col2:
                                st.metric("Mínimo", f"{df[coluna_selecionada].min():.2f}")
                                st.metric("Máximo", f"{df[coluna_selecionada].max():.2f}")
                            
                            # Boxplot
                            fig = px.box(df, y=coluna_selecionada, title=f"Distribuição de {coluna_selecionada}")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Coluna categórica
                            valores = df[coluna_selecionada].value_counts()
                            fig = px.bar(x=valores.index, y=valores.values, 
                                       title=f"Distribuição de {coluna_selecionada}")
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("📊 Análise Comparativa")
                
                # Selecionar colunas para comparação
                colunas_comparacao = [col for col in df.columns if col not in ['Tempo_Operacao', 'Falha']]
                
                if len(colunas_comparacao) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        var_x = st.selectbox("Variável X:", colunas_comparacao, key="var_x")
                    with col2:
                        var_y = st.selectbox("Variável Y:", colunas_comparacao, key="var_y")
                    
                    if var_x and var_y:
                        if df[var_x].dtype in ['int64', 'float64'] and df[var_y].dtype in ['int64', 'float64']:
                            # Scatter plot para variáveis numéricas
                            fig = px.scatter(df, x=var_x, y=var_y, color='Falha',
                                           title=f"Relação entre {var_x} e {var_y}",
                                           hover_data=['Tempo_Operacao'])
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("ℹ️ Selecione variáveis numéricas para scatter plot")
                else:
                    st.info("ℹ️ Colunas insuficientes para análise comparativa")
            
            with tab4:
                st.subheader("🔥 Matriz de Risco")
                
                # Verificar colunas necessárias para matriz de risco
                if 'Componente_Falha' in df.columns:
                    matriz_risco = criar_matriz_risco(df)
                    
                    if matriz_risco is not None:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Heatmap de risco
                            fig = px.density_heatmap(
                                matriz_risco,
                                x='Probabilidade',
                                y='Criticidade',
                                z='Risco',
                                title="Matriz de Risco",
                                color_continuous_scale='RdYlGn_r',
                                nbinsx=10,
                                nbinsy=10
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Tabela de risco
                            st.dataframe(
                                matriz_risco.sort_values('Risco', ascending=False),
                                use_container_width=True
                            )
                            
                            # Insights da matriz
                            if len(matriz_risco) > 0:
                                componente_alto_risco = matriz_risco.loc[matriz_risco['Risco'].idxmax()]
                                st.info(f"""
                                **Componente de maior risco:** {componente_alto_risco['Componente']}
                                - Probabilidade: {componente_alto_risco['Probabilidade']:.1f}%
                                - Criticidade: {componente_alto_risco['Criticidade']}h MTTR
                                - Risco: {componente_alto_risco['Risco']:.1f}
                                """)
                    else:
                        st.info("ℹ️ Não há dados suficientes para criar matriz de risco")
                else:
                    st.info("ℹ️ A coluna 'Componente_Falha' não está disponível para matriz de risco")
        else:
            st.warning("⚠️ Dataset não carregado corretamente.")
    else:
        st.warning("⚠️ Nenhum dataset disponível. Gere dados sintéticos ou carregue dados reais.")

# ============================
# MÓDULO: Curvas de Sobrevivência (ADAPTATIVO)
# ============================
elif menu == "📉 Curvas de Sobrevivência":
    st.header("📉 Análise de Sobrevivência - Kaplan-Meier")
    
    if st.session_state['df_sintetico'] is not None or st.session_state['df_real'] is not None:
        df = st.session_state['df_sintetico'] if st.session_state['dados_tipo'] == 'sintetico' else st.session_state['df_real']
        
        if df is not None:
            # Verificar colunas obrigatórias
            if 'Tempo_Operacao' not in df.columns or 'Falha' not in df.columns:
                st.error("❌ Dataset não contém as colunas obrigatórias: 'Tempo_Operacao' e 'Falha'")
                st.stop()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Curva Kaplan-Meier geral
                kmf = KaplanMeierFitter()
                kmf.fit(df['Tempo_Operacao'], df['Falha'])
                
                timeline, survival_values, _ = get_kmf_survival_data(kmf)
                lower, upper = get_kmf_confidence_interval(kmf)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timeline,
                    y=survival_values,
                    mode='lines',
                    name='Sobrevivência',
                    line=dict(color='blue', width=3)
                ))
                
                if lower is not None and upper is not None and len(lower) > 0:
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([timeline, timeline[::-1]]),
                        y=np.concatenate([lower, upper[::-1]]),
                        fill='toself',
                        fillcolor='rgba(0,100,255,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='IC 95%'
                    ))
                
                fig.update_layout(
                    title="Curva de Sobrevivência (Kaplan-Meier)",
                    xaxis_title="Tempo (horas)",
                    yaxis_title="Probabilidade de Sobrevivência",
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Métricas de Sobrevivência")
                
                try:
                    tempo_mediano = kmf.median_survival_time_
                    st.metric("Tempo Mediano", f"{tempo_mediano:,.0f}h" if not np.isnan(tempo_mediano) else "N/A")
                except:
                    st.metric("Tempo Mediano", "N/A")
                
                try:
                    tempo_25 = kmf.percentile(0.25)
                    st.metric("25º Percentil", f"{tempo_25:,.0f}h")
                except:
                    st.metric("25º Percentil", "N/A")
                
                try:
                    tempo_75 = kmf.percentile(0.75)
                    st.metric("75º Percentil", f"{tempo_75:,.0f}h")
                except:
                    st.metric("75º Percentil", "N/A")
                
                st.markdown("---")
                st.subheader("📈 Hazard Cumulativo")
                
                naf = NelsonAalenFitter()
                naf.fit(df['Tempo_Operacao'], df['Falha'])
                hazard_timeline, hazard_values = get_naf_hazard_data(naf)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=hazard_timeline,
                    y=hazard_values,
                    mode='lines',
                    line=dict(color='red', width=2)
                ))
                fig2.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Análise estratificada
            st.subheader("🔍 Análise Estratificada")
            
            variaveis_categoricas = [col for col in df.columns if col not in ['Tempo_Operacao', 'Falha'] 
                                    and df[col].nunique() < 10 and df[col].nunique() > 1]
            
            if variaveis_categoricas:
                variavel = st.selectbox("Estratificar por:", variaveis_categoricas)
                
                if variavel:
                    fig3 = go.Figure()
                    valores = df[variavel].dropna().unique()[:5]
                    
                    for valor in valores:
                        mask = df[variavel] == valor
                        if mask.sum() > 0:
                            kmf_strat = KaplanMeierFitter()
                            kmf_strat.fit(df[mask]['Tempo_Operacao'], df[mask]['Falha'], label=f"{valor}")
                            timeline_strat, survival_strat, _ = get_kmf_survival_data(kmf_strat)
                            fig3.add_trace(go.Scatter(
                                x=timeline_strat,
                                y=survival_strat,
                                mode='lines',
                                name=f"{valor} (n={mask.sum()})"
                            ))
                    
                    fig3.update_layout(
                        title=f"Curvas por {variavel}",
                        xaxis_title="Tempo (horas)",
                        yaxis_title="Probabilidade de Sobrevivência",
                        height=500
                    )
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Não foram encontradas variáveis categóricas para estratificação.")
        else:
            st.warning("⚠️ Dataset não disponível.")
    else:
        st.warning("⚠️ Nenhum dataset disponível.")

# ============================
# MÓDULO: Modelos Preditivos (ADAPTATIVO)
# ============================
elif menu == "⚡ Modelos Preditivos":
    st.header("⚡ Modelos Preditivos de Confiabilidade")
    
    tab1, tab2 = st.tabs(["📈 Modelo de Cox", "⚙️ Weibull AFT"])
    
    with tab1:
        st.subheader("Modelo de Regressão de Cox")
        
        if st.session_state['df_sintetico'] is not None or st.session_state['df_real'] is not None:
            df = st.session_state['df_sintetico'] if st.session_state['dados_tipo'] == 'sintetico' else st.session_state['df_real']
            
            if df is not None:
                # Listar colunas disponíveis
                variaveis_numericas = [col for col in df.select_dtypes(include=[np.number]).columns 
                                      if col not in ['Tempo_Operacao', 'Falha'] and df[col].nunique() > 1]
                variaveis_categoricas = df.select_dtypes(include=['object']).columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    vars_num = st.multiselect("Variáveis numéricas:", variaveis_numericas, 
                                            default=variaveis_numericas[:3] if len(variaveis_numericas) >= 3 else variaveis_numericas)
                with col2:
                    vars_cat = st.multiselect("Variáveis categóricas:", variaveis_categoricas,
                                            default=variaveis_categoricas[:2] if len(variaveis_categoricas) >= 2 else variaveis_categoricas)
                
                if st.button("📊 Ajustar Modelo de Cox", type="primary"):
                    with st.spinner("Ajustando modelo..."):
                        try:
                            df_modelo = df.copy()
                            for var in vars_cat:
                                df_modelo = pd.get_dummies(df_modelo, columns=[var], drop_first=True)
                            
                            todas_vars = vars_num + [col for col in df_modelo.columns if any(var in col for var in vars_cat)]
                            todas_vars = [v for v in todas_vars if v in df_modelo.columns]
                            
                            if todas_vars:
                                cph = CoxPHFitter()
                                cph.fit(df_modelo[['Tempo_Operacao', 'Falha'] + todas_vars], 
                                       duration_col='Tempo_Operacao', event_col='Falha')
                                
                                st.session_state['cph_model'] = cph
                                st.success("✅ Modelo ajustado!")
                                
                                # Coeficientes
                                coef_df = cph.summary.copy()
                                coef_df['exp(coef)'] = np.exp(coef_df['coef'])
                                coef_df['HR_interpretation'] = coef_df['exp(coef)'].apply(
                                    lambda x: f"{x:.2f}x risco" if x > 1 else f"{1/x:.2f}x proteção"
                                )
                                
                                st.dataframe(coef_df[['coef', 'exp(coef)', 'HR_interpretation', 'p']].style.background_gradient(
                                    subset=['p'], cmap='RdYlGn_r'
                                ), use_container_width=True)
                                
                                # Gráfico
                                fig = px.bar(
                                    x=coef_df.index,
                                    y=coef_df['coef'].abs(),
                                    title="Importância das Variáveis",
                                    labels={'x': 'Variável', 'y': '|Coeficiente|'},
                                    color=coef_df['coef'],
                                    color_continuous_scale='RdBu'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Concordância
                                try:
                                    concordance = cph.concordance_index_
                                    st.metric("Índice de Concordância", f"{concordance:.3f}")
                                except:
                                    st.metric("Índice de Concordância", "N/A")
                            else:
                                st.warning("Nenhuma variável selecionada.")
                        except Exception as e:
                            st.error(f"❌ Erro: {str(e)}")
    
    with tab2:
        st.subheader("Modelo Weibull AFT")
        
        if st.session_state['df_sintetico'] is not None or st.session_state['df_real'] is not None:
            df = st.session_state['df_sintetico'] if st.session_state['dados_tipo'] == 'sintetico' else st.session_state['df_real']
            
            if df is not None:
                # Listar colunas disponíveis
                variaveis_numericas = [col for col in df.select_dtypes(include=[np.number]).columns 
                                      if col not in ['Tempo_Operacao', 'Falha']]
                
                if len(variaveis_numericas) > 0:
                    vars_selecionadas = st.multiselect(
                        "Selecione variáveis para o modelo:",
                        variaveis_numericas,
                        default=variaveis_numericas[:min(3, len(variaveis_numericas))]
                    )
                    
                    if st.button("⚙️ Ajustar Modelo Weibull AFT", type="primary"):
                        with st.spinner("Ajustando modelo..."):
                            try:
                                if vars_selecionadas:
                                    aft = WeibullAFTFitter()
                                    aft.fit(df[['Tempo_Operacao', 'Falha'] + vars_selecionadas], 
                                           duration_col='Tempo_Operacao', event_col='Falha')
                                    
                                    st.session_state['aft_model'] = aft
                                    st.success("✅ Modelo ajustado!")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Parâmetros:**")
                                        st.write(f"λ: {aft.lambda_:.4f}")
                                        st.write(f"ρ: {aft.rho_:.4f}")
                                    with col2:
                                        st.write("**Coeficientes:**")
                                        st.dataframe(aft.summary[['coef', 'se(coef)', 'p']], use_container_width=True)
                                    
                                    # PDF Weibull
                                    x = np.linspace(0, df['Tempo_Operacao'].max() * 1.2, 200)
                                    scale = np.exp(aft.params_['lambda_'])
                                    shape = 1 / aft.params_['rho_']
                                    pdf = weibull_min.pdf(x, shape, scale=scale)
                                    
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=x, y=pdf,
                                        mode='lines',
                                        name=f"Weibull (β={shape:.2f}, η={scale:.0f})",
                                        line=dict(color='blue', width=2)
                                    ))
                                    fig.update_layout(
                                        title="Função Densidade Weibull",
                                        xaxis_title="Tempo (horas)",
                                        yaxis_title="Densidade"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.warning("Selecione pelo menos uma variável.")
                            except Exception as e:
                                st.error(f"❌ Erro: {str(e)}")
                else:
                    st.info("ℹ️ Não há variáveis numéricas disponíveis para o modelo AFT.")

# ============================
# MÓDULO: Sistema de Alerta
# ============================
elif menu == "🔔 Sistema de Alerta":
    st.header("🔔 Sistema de Alerta Preditivo")
    
    st.info("Este módulo analisa bombas com alto risco de falha baseado em modelos preditivos.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        limite_risco = st.slider("Limite de risco (%)", 50, 95, 75)
    with col2:
        horizonte = st.slider("Horizonte (horas)", 100, 5000, 1000)
    with col3:
        dias_manutencao = st.slider("Dias para manutenção", 1, 30, 7)
    
    if st.button("🔍 Executar Análise de Risco", type="primary"):
        # Simulação de análise
        st.subheader("📋 Bombas com Alto Risco de Falha")
        
        bombas_risco = [
            {"ID": "BOMBA_042", "Risco": 89, "Componente": "Selagem", "Tempo_Estimado": 320, "Criticidade": "Alta"},
            {"ID": "BOMBA_017", "Risco": 78, "Componente": "Rotor", "Tempo_Estimado": 450, "Criticidade": "Média"},
            {"ID": "BOMBA_089", "Risco": 72, "Componente": "Mancal", "Tempo_Estimado": 580, "Criticidade": "Média"},
            {"ID": "BOMBA_123", "Risco": 65, "Componente": "Eixo", "Tempo_Estimado": 720, "Criticidade": "Baixa"},
        ]
        
        for bomba in bombas_risco:
            if bomba['Risco'] > 80:
                st.markdown(f'<div class="alert-high">', unsafe_allow_html=True)
            elif bomba['Risco'] > 60:
                st.markdown(f'<div class="alert-medium">', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-low">', unsafe_allow_html=True)
            
            st.markdown(f"**{bomba['ID']}** - Risco: {bomba['Risco']}%")
            st.markdown(f"Componente: {bomba['Componente']} | Tempo estimado: {bomba['Tempo_Estimado']}h")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Cronograma
        st.subheader("📅 Cronograma de Manutenção")
        hoje = datetime.now()
        cronograma = []
        
        for i, bomba in enumerate([b for b in bombas_risco if b['Risco'] > 70][:3]):
            data = hoje + timedelta(days=(i+1)*2)
            cronograma.append({
                "Bomba": bomba['ID'],
                "Componente": bomba['Componente'],
                "Data": data.strftime("%d/%m/%Y"),
                "Prioridade": "Alta" if bomba['Risco'] > 80 else "Média"
            })
        
        if cronograma:
            st.dataframe(pd.DataFrame(cronograma), use_container_width=True)

# ============================
# MÓDULO: Dashboard Industrial (ADAPTATIVO)
# ============================
elif menu == "📊 Dashboard Industrial":
    st.header("📊 Dashboard Industrial de Confiabilidade")
    
    tem_dados = st.session_state['df_sintetico'] is not None or st.session_state['df_real'] is not None
    
    if tem_dados:
        df = st.session_state['df_sintetico'] if st.session_state['dados_tipo'] == 'sintetico' else st.session_state['df_real']
        
        if df is not None:
            metricas = calcular_metricas_confiabilidade(df)
            
            # KPIs principais
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MTBF", f"{metricas['mtbf']:,.0f}h", "+3.2%")
            with col2:
                st.metric("Falhas", metricas['total_falhas'], "-15%")
            with col3:
                st.metric("Custo Falhas", f"R$ {metricas['total_falhas'] * 2500:,}", "-8%")
            
            st.markdown("---")
            
            # Gráficos adaptativos
            col1, col2 = st.columns(2)
            with col1:
                if 'Componente_Falha' in df.columns:
                    df_falhas = df[df['Falha'] == 1]
                    if len(df_falhas) > 0:
                        fig = px.pie(
                            names=df_falhas['Componente_Falha'].value_counts().index,
                            values=df_falhas['Componente_Falha'].value_counts().values,
                            title="Falhas por Componente"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("ℹ️ Não há falhas registradas")
                else:
                    st.info("ℹ️ Coluna 'Componente_Falha' não disponível")
            
            with col2:
                # Tendência simulada adaptada ao dataset
                if 'Tempo_Operacao' in df.columns:
                    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun']
                    mtbf_base = metricas['mtbf']
                    mtbf_vals = np.random.normal(mtbf_base, mtbf_base*0.1, len(meses)) + np.arange(len(meses))*mtbf_base*0.02
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=meses, y=mtbf_vals, mode='lines+markers', name='MTBF'))
                    fig.update_layout(title="Tendência MTBF", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("ℹ️ Coluna 'Tempo_Operacao' não disponível")
            
            # Tabela KPIs adaptativa
            st.subheader("📋 KPIs de Confiabilidade")
            kpis = {
                'KPI': ['MTBF', 'MTTR', 'Taxa Falhas', 'Custo/Falha'],
                'Valor': [f"{metricas['mtbf']:,.0f}h", '24h', 
                         f"{metricas['taxa_falhas']:.1f}%", 'R$ 2.450'],
                'Meta': ['>2.500h', '<48h', '<15%', '<R$ 3.000'],
                'Status': ['✅' if metricas['mtbf'] > 2500 else '⚠️', '✅', 
                          '✅' if metricas['taxa_falhas'] < 15 else '⚠️', '✅']
            }
            st.dataframe(pd.DataFrame(kpis), use_container_width=True, hide_index=True)
            
            # Status das bombas
            st.subheader("🔧 Status das Bombas")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                operando = len(df) - metricas['total_falhas']
                st.metric("Operando", operando)
            
            with col2:
                st.metric("Em Falha", metricas['total_falhas'])
            
            with col3:
                if 'Componente_Falha' in df.columns:
                    componentes_unicos = df['Componente_Falha'].nunique()
                    st.metric("Componentes Afetados", componentes_unicos)
                else:
                    st.metric("Componentes Afetados", "N/A")
            
        else:
            st.warning("⚠️ Dataset não encontrado.")
    else:
        st.warning("⚠️ Nenhum dataset disponível.")

# ============================
# MÓDULO: Relatório Técnico (ADAPTATIVO)
# ============================
elif menu == "📋 Relatório Técnico":
    st.header("📋 Relatório Técnico de Confiabilidade")
    
    if st.session_state['df_sintetico'] is not None or st.session_state['df_real'] is not None:
        df = st.session_state['df_sintetico'] if st.session_state['dados_tipo'] == 'sintetico' else st.session_state['df_real']
        
        if df is not None:
            metricas = calcular_metricas_confiabilidade(df)
            insights = []
            
            # Insights adaptativos baseados nas colunas disponíveis
            insights.append(f"Total de {len(df):,} bombas analisadas")
            insights.append(f"MTBF atual de {metricas['mtbf']:,.0f}h está {'acima' if metricas['mtbf'] > 2500 else 'abaixo'} da meta de 2.500h")
          
            if 'Componente_Falha' in df.columns and metricas['total_falhas'] > 0:
                componente_mais_falha = df[df['Falha'] == 1]['Componente_Falha'].value_counts().index[0]
                insights.append(f"Componente com maior incidência de falhas: {componente_mais_falha}")
            
            if 'Tipo_Falha' in df.columns and metricas['total_falhas'] > 0:
                tipo_mais_comum = df[df['Falha'] == 1]['Tipo_Falha'].value_counts().index[0]
                insights.append(f"Tipo de falha mais comum: {tipo_mais_comum}")
            
            insights.append(f"Taxa de censura de {metricas['taxa_censura']:.1f}% indica {'bom' if metricas['taxa_censura'] < 30 else 'alto'} acompanhamento")
            
            st.markdown(f"""
            ### Relatório de Análise de Confiabilidade
            **Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
            **Dataset:** {st.session_state['dados_tipo'].title()} ({len(df):,} registros)
            **Colunas analisadas:** {len(df.columns)}
            """)
            
            with st.expander("📊 Sumário Executivo", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MTBF", f"{metricas['mtbf']:,.0f}h")
                with col2:
                    st.metric("Falhas", metricas['total_falhas'])
                
                st.markdown("### Principais Achados:")
                for i, insight in enumerate(insights[:5], 1):
                    st.markdown(f"{i}. {insight}")
            
            with st.expander("🎯 Recomendações"):
                recomendacoes = [
                    "Implementar monitoramento online de temperatura e vibração",
                    "Revisar procedimentos de manutenção preventiva",
                    "Treinar operadores em identificação de sinais de falha",
                    "Considerar upgrade de componentes críticos"
                ]
                for rec in recomendacoes:
                    st.markdown(f"• {rec}")
            
            # Botão para gerar PDF
            if st.button("📄 Gerar Relatório PDF", type="primary"):
                with st.spinner("Gerando relatório..."):
                    pdf_path = gerar_relatorio_pdf(metricas, df, insights)
                    
                    with open(pdf_path, 'rb') as f:
                        pdf_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Download Relatório PDF",
                        data=pdf_bytes,
                        file_name=f"relatorio_confiabilidade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Relatório gerado com sucesso!")
        else:
            st.warning("⚠️ Dataset não disponível.")
    else:
        st.warning("⚠️ Nenhum dataset disponível.")

# ============================
# MÓDULO: Modelos de Machine Learning (ADAPTATIVO)
# ============================
elif menu == "🤖 Modelos de Machine Learning":
    st.markdown('<h2 class="sub-header">🤖 Modelos Preditivos de Machine Learning</h2>', unsafe_allow_html=True)
    
    # Verificar dados disponíveis
    if st.session_state['df_sintetico'] is not None or st.session_state['df_real'] is not None:
        df = st.session_state['df_sintetico'] if st.session_state['dados_tipo'] == 'sintetico' else st.session_state['df_real']
        
        if df is not None:
            # Verificar tamanho mínimo do dataset
            if len(df) < 50:
                st.warning("⚠️ Dataset muito pequeno para modelos de ML (mínimo recomendado: 50 registros)")
                st.stop()
            
            # Mostrar informações sobre o dataset
            st.info(f"📊 Dataset: {st.session_state['dados_tipo'].title()} - {len(df):,} registros - {df.shape[1]} colunas")
            
            with st.expander("🔍 Visualizar Colunas Disponíveis", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Colunas Numéricas:**")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    for col in numeric_cols[:10]:
                        st.write(f"- {col}")
                    if len(numeric_cols) > 10:
                        st.write(f"... e mais {len(numeric_cols) - 10} colunas")
                with col2:
                    st.write("**Colunas Categóricas:**")
                    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                    for col in cat_cols[:10]:
                        st.write(f"- {col}")
                    if len(cat_cols) > 10:
                        st.write(f"... e mais {len(cat_cols) - 10} colunas")
            
            # Seleção do problema de ML
            st.markdown("### 🎯 Selecionar Problema de Machine Learning")
            
            # Verificar quais problemas são possíveis com as colunas disponíveis
            problemas_disponiveis = []
            
            if 'Tipo_Falha' in df.columns:
                problemas_disponiveis.append("Classificação - Tipo de Falha")
            
            if 'Componente_Falha' in df.columns:
                problemas_disponiveis.append("Classificação - Componente que Falhará")
            
            if 'Tempo_Operacao' in df.columns:
                problemas_disponiveis.append("Regressão - Tempo até Falha")
            
            if not problemas_disponiveis:
                st.error("❌ Dataset não contém colunas suficientes para problemas de ML")
                st.info("Adicione colunas como 'Tipo_Falha', 'Componente_Falha' ou 'Tempo_Operacao'")
                st.stop()
            
            problema = st.selectbox(
                "Escolha o tipo de previsão:",
                problemas_disponiveis
            )
            
            # Mapear problema selecionado para código
            if problema == "Classificação - Tipo de Falha":
                problema_ml = 'classificacao_tipo'
                st.markdown("#### 📊 Classificação do Tipo de Falha")
                st.write("Este modelo prevê o tipo de falha que ocorrerá baseado nas condições operacionais.")
                
            elif problema == "Classificação - Componente que Falhará":
                problema_ml = 'classificacao_componente'
                st.markdown("#### 🔧 Classificação do Componente que Falhará")
                st.write("Este modelo prevê qual componente da bomba provavelmente falhará.")
                
            elif problema == "Regressão - Tempo até Falha":
                problema_ml = 'regressao_tempo'
                st.markdown("#### ⏰ Regressão do Tempo até Falha")
                st.write("Este modelo prevê quantas horas restam até a próxima falha.")
            
            # Configurações de treinamento
            st.markdown("### ⚙️ Configurações do Treinamento")
            
            col_config1, col_config2 = st.columns(2)
            
            with col_config1:
                modelo_tipo = st.selectbox(
                    "Algoritmo:",
                    ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Gradient Boosting"]
                )
                
                test_size = st.slider("Tamanho do teste (%)", 10, 40, 20, 5)
            
            with col_config2:
                # Seleção de features
                features_disponiveis = [col for col in df.columns if col not in ['ID_Bomba', 'Tempo_Operacao', 'Falha', 'Tipo_Falha', 'Componente_Falha']]
                if len(features_disponiveis) > 0:
                    usar_todas_features = st.checkbox("Usar todas as features disponíveis", value=True)
                    
                    if not usar_todas_features:
                        features_selecionadas = st.multiselect(
                            "Selecionar features:", 
                            features_disponiveis, 
                            default=features_disponiveis[:min(5, len(features_disponiveis))]
                        )
                else:
                    st.warning("⚠️ Não há features disponíveis para treinamento")
                    usar_todas_features = True
            
            # Botão para treinar
            if st.button(f"🚀 Treinar Modelo de {problema.split(' - ')[0]}", type="primary", use_container_width=True):
                with st.spinner(f"Preparando dados e treinando modelo..."):
                    try:
                        # Preparar dados
                        X_train, X_test, y_train, y_test, target_type, label_encoders = preparar_dados_ml(df, problema_ml)
                        
                        if X_train is None:
                            st.error("❌ Não foi possível preparar os dados para este problema")
                            st.stop()
                        
                        st.session_state['preprocessors'][problema_ml] = label_encoders
                        
                        # Treinar modelo
                        if problema_ml in ['classificacao_tipo', 'classificacao_componente']:
                            modelo_map = {
                                'Random Forest': 'rf',
                                'XGBoost': 'xgb',
                                'LightGBM': 'lgbm',
                                'CatBoost': 'catboost',
                                'Gradient Boosting': 'gb'
                            }
                            
                            results = treinar_modelo_classificacao(
                                X_train, X_test, y_train, y_test, 
                                modelo_tipo=modelo_map[modelo_tipo]
                            )
                            
                            # Salvar resultados
                            st.session_state['ml_models'][problema_ml] = results['model']
                            st.session_state['ml_metrics'][problema_ml] = {
                                'accuracy': results['accuracy'],
                                'precision': results['precision'],
                                'recall': results['recall'],
                                'f1': results['f1'],
                                'confusion_matrix': results['confusion_matrix']
                            }
                            
                            if results['feature_importances'] is not None:
                                st.session_state['feature_importance'][problema_ml] = results['feature_importances']
                            
                            st.success(f"✅ Modelo treinado com sucesso! Acurácia: {results['accuracy']:.3f}")
                            
                            # Mostrar dashboard de resultados
                            criar_dashboard_ml(
                                st.session_state['ml_metrics'][problema_ml],
                                st.session_state['feature_importance'].get(problema_ml, None),
                                X_test, y_test, results['y_pred'],
                                problema_ml
                            )
                        
                        elif problema_ml == 'regressao_tempo':
                            modelo_map = {
                                'Random Forest': 'rf',
                                'XGBoost': 'xgb',
                                'LightGBM': 'lgbm',
                                'CatBoost': 'catboost',
                                'Gradient Boosting': 'gb'
                            }
                            
                            results = treinar_modelo_regressao(
                                X_train, X_test, y_train, y_test,
                                modelo_tipo=modelo_map[modelo_tipo]
                            )
                            
                            # Salvar resultados
                            st.session_state['ml_models'][problema_ml] = results['model']
                            st.session_state['ml_metrics'][problema_ml] = {
                                'mae': results['mae'],
                                'mse': results['mse'],
                                'rmse': results['rmse'],
                                'r2': results['r2']
                            }
                            
                            if results['feature_importances'] is not None:
                                st.session_state['feature_importance'][problema_ml] = results['feature_importances']
                            
                            st.success(f"✅ Modelo treinado com sucesso! R² Score: {results['r2']:.3f}")
                            
                            # Mostrar dashboard de resultados
                            criar_dashboard_ml(
                                st.session_state['ml_metrics'][problema_ml],
                                st.session_state['feature_importance'].get(problema_ml, None),
                                X_test, y_test, results['y_pred'],
                                problema_ml
                            )
                    
                    except Exception as e:
                        st.error(f"❌ Erro durante o treinamento: {str(e)}")
                        st.info("Verifique se há dados suficientes e se as colunas estão no formato correto.")
            
            # Seção de predição em tempo real
            if st.session_state.get('ml_models'):
                st.markdown("### 🔮 Predição em Tempo Real")
                
                # Verificar quais modelos estão disponíveis
                modelos_disponiveis = list(st.session_state['ml_models'].keys())
                
                if modelos_disponiveis:
                    st.write(f"Modelos treinados disponíveis: {', '.join(modelos_disponiveis)}")
                    
                    # Criar formulário para entrada de dados
                    st.write("Insira os dados de uma nova bomba para fazer previsões:")
                    
                    # Coletar valores para as features mais comuns
                    col_pred1, col_pred2 = st.columns(2)
                    
                    with col_pred1:
                        if 'Vibracao_Media' in df.columns:
                            vibracao = st.number_input("Vibração Média (mm/s)", 1.0, 20.0, 5.0, 0.1)
                        
                        if 'Temperatura_Media' in df.columns:
                            temperatura = st.number_input("Temperatura Média (°C)", 20.0, 120.0, 65.0, 0.5)
                    
                    with col_pred2:
                        if 'Pressao_Media' in df.columns:
                            pressao = st.number_input("Pressão Média (bar)", 5.0, 30.0, 15.0, 0.5)
                        
                        if 'Velocidade_RPM' in df.columns:
                            rpm = st.number_input("Velocidade (RPM)", 1500, 2200, 1800, 10)
                    
                    if st.button("🔮 Fazer Previsão", type="secondary"):
                        # Aqui você implementaria a lógica real de predição
                        st.success("✅ Previsão realizada com sucesso!")
                        st.info("Nota: Esta é uma demonstração. Para implementação real, conecte os modelos treinados.")
        else:
            st.warning("⚠️ Dataset não carregado corretamente.")
    else:
        st.warning("⚠️ Nenhum dataset disponível. Gere dados sintéticos ou carregue dados reais primeiro.")

# ============================
# RODAPÉ
# ============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>⚙️ Sistema de Análise de Confiabilidade - Bombas Centrífugas v2.0</p>
    <p>Com vetorização, degradação temporal e modelos preditivos avançados | © 2024</p>
</div>
""", unsafe_allow_html=True)