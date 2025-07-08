# preprocessing.py
# Funções para carregar, limpar, pré-processar e dividir os dados do câncer de mama
# Explicações detalhadas para cada etapa do pipeline de dados

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Caminho do arquivo CSV com os dados
CSV_PATH = 'data/data.csv'

def load_and_clean_data():
    """
    Lê e limpa o dataset.
    Remove colunas irrelevantes e vazias, como 'Unnamed: 32' e 'id'.
    Retorna um DataFrame limpo.
    """
    df = pd.read_csv(CSV_PATH)

    # Remove colunas desnecessárias e vazias
    colunas_remover = ['Unnamed: 32', 'id']
    df.drop(columns=[col for col in colunas_remover if col in df.columns], inplace=True)

    return df

def preprocess_data(df, target_column='diagnosis'):
    """
    Codifica a coluna alvo ('diagnosis') para valores numéricos (0 e 1) e normaliza os dados de entrada.
    Retorna os dados prontos para o modelo.
    """
    X = df.drop(columns=[target_column])  # Dados de entrada (features)
    y = df[target_column]  # Alvo (diagnóstico)

    le = LabelEncoder()  # Codificador para transformar 'B'/'M' em 0/1
    y_encoded = le.fit_transform(y)

    scaler = StandardScaler()  # Normaliza os dados para média 0 e desvio 1
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divide os dados em conjuntos de treino e teste.
    test_size: porcentagem dos dados para teste (padrão 20%).
    random_state: semente para reprodutibilidade.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)