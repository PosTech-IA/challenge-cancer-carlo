# main.py
# Pipeline principal para análise e predição de câncer de mama
# Este script executa todas as etapas: carregamento, pré-processamento, treinamento, avaliação, interpretação e salvamento dos modelos.
# Cada etapa é explicada detalhadamente nos comentários abaixo.
#
# Autor: [Seu Nome]
# Data: [Data de modificação]
from eda import run_cancer_eda
from preprocessing import load_and_clean_data, preprocess_data, split_data  # Funções para preparar os dados
from train import train_all_models  # Função para treinar modelos padrão
from evaluate import evaluate_models, plot_feature_importance, explain_with_shap  # Funções para avaliar e interpretar modelos
from optimize import optimize_logistic_regression, optimize_decision_tree  # Funções para otimizar modelos
import pandas as pd
import joblib  # Para salvar e carregar modelos treinados
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from train_top_features import get_top_n_features, filter_features

def main():
    print("🚀 INICIANDO PIPELINE...\n")

    # Etapa 0: Análise exploratória de dados (EDA)
    print("\n📊 Gerando gráficos de EDA...")
    df_original = pd.read_csv("data/data.csv").drop(columns=["Unnamed: 32", "id"], errors="ignore")
    run_cancer_eda(df_original)
    print("📁 Gráficos salvos em: analysis/")

    # Etapa 1: Carregamento e pré-processamento dos dados
    # Lê o arquivo CSV, remove colunas desnecessárias, codifica e normaliza os dados
    df = load_and_clean_data()

    # Etapa 0: Análise exploratória de dados (EDA)
    print("\n📊 Gerando gráficos de EDA...")
    df_original = pd.read_csv("data/data.csv").drop(columns=["Unnamed: 32", "id"], errors="ignore")
    run_cancer_eda(df_original)
    print("📁 Gráficos salvos em: analysis/")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"✔️ Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")


    # Etapa 2: Treinamento dos modelos padrão (sem otimização de hiperparâmetros)
    print("\n🤖 TREINANDO MODELOS PADRÃO...")
    models_default = train_all_models(X_train, y_train)

    # Etapa 3: Treinamento dos modelos otimizados (com busca de melhores hiperparâmetros)
    print("\n🔧 OTIMIZANDO MODELOS COM VALIDAÇÃO CRUZADA...")
    models_optimized = {
        "Logistic Regression": optimize_logistic_regression(X_train, y_train),
        "Decision Tree": optimize_decision_tree(X_train, y_train)
    }

    # Etapa 4: Avaliação dos modelos padrão e otimizados
    print("\n📊 AVALIANDO MODELOS PADRÃO...")
    evaluate_models(models_default, X_test, y_test)

    print("\n📊 AVALIANDO MODELOS OTIMIZADOS...")
    evaluate_models(models_optimized, X_test, y_test)

    # Etapa 5: Interpretação da importância das variáveis (feature importance) para a árvore otimizada
    print("\n🌿 INTERPRETANDO ÁRVORE DE DECISÃO (OTIMIZADA)...")
    # Lê os nomes das colunas do arquivo original para exibir no gráfico
    feature_names = pd.read_csv('data/data.csv').drop(columns=['Unnamed: 32', 'id', 'diagnosis']).columns
    plot_feature_importance(models_optimized["Decision Tree"], feature_names, model_name="Decision_Tree_Optimizada")

    # Etapa 6: Explicação do modelo de regressão logística otimizada usando SHAP
    print("\n🧠 Explicando Regressão Logística (otimizada) com SHAP...")
    explain_with_shap(models_optimized["Logistic Regression"], X_train, feature_names, model_name="Regressao_Logistica_Otimizada")

    print("\n✅ Pipeline finalizado com sucesso.")

    # Etapa 7: Salvar os modelos otimizados em arquivos para uso futuro
    print("\n💾 Salvando modelos otimizados...")
    os.makedirs("outputs/models", exist_ok=True)  # Cria a pasta se não existir
    joblib.dump(models_optimized["Logistic Regression"], "outputs/models/regression_optimized.pkl")
    joblib.dump(models_optimized["Decision Tree"], "outputs/models/tree_optimized.pkl")
    print("📁 Modelos salvos em: outputs/models/")

    print("\n✅ Pipeline finalizado com sucesso.")

    # Etapa opcional: testar performance com as top-N features da Árvore de Decisão
    print("\n🧪 Testando modelo com as 10 variáveis mais importantes...")

    # Recupera nomes das colunas
    feature_names = pd.read_csv('data/data.csv').drop(columns=['Unnamed: 32', 'id', 'diagnosis']).columns

    # Treina modelo completo com todas as features para pegar importâncias
    full_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    top_10_features = get_top_n_features(full_tree, feature_names, n=4)
    print("🏆 Top 10 features:", top_10_features)

    # Filtra os dados de treino e teste com as top 10
    X_train_top = filter_features(X_train, feature_names, top_10_features)
    X_test_top = filter_features(X_test, feature_names, top_10_features)

    # Treina novo modelo com top 10
    model_top = DecisionTreeClassifier(random_state=42)
    model_top.fit(X_train_top, y_train)
    y_pred_top = model_top.predict(X_test_top)

    # Avalia o desempenho
    print("\n📊 Avaliação com Top-10 Features")
    print("Accuracy:", accuracy_score(y_test, y_pred_top))
    print("Recall:", recall_score(y_test, y_pred_top))
    print("F1-score:", f1_score(y_test, y_pred_top))
    print("\nClassification Report:\n", classification_report(y_test, y_pred_top))

if __name__ == "__main__":
    main()