# explain_single.py
# Gera explicação SHAP para um único exame de câncer de mama
# Explicações detalhadas para cada etapa

import pandas as pd
import shap
import joblib
from preprocessing import preprocess_data
import os

# Caminho do modelo treinado (regressão logística otimizada)
modelo_path = "outputs/models/regression_optimized.pkl"
model = joblib.load(modelo_path)  # Carrega o modelo salvo

# Caminho do exame individual (novo exame a ser explicado)
csv_path = "data/exame_novo.csv"  # Arquivo deve ter as mesmas colunas do treino

# Lê os dados do novo exame
df = pd.read_csv(csv_path)

# Adiciona coluna fictícia 'diagnosis' para compatibilizar com o pipeline de pré-processamento
# (o valor não será usado, mas é necessário para a função preprocess_data)
df.insert(0, "diagnosis", "B")

# Pré-processamento (usa o mesmo padrão do pipeline principal)
X_scaled, _ = preprocess_data(df)

# Recupera nomes das colunas originais (sem a coluna 'diagnosis')
feature_names = df.drop(columns=["diagnosis"]).columns
X_df = pd.DataFrame(X_scaled, columns=feature_names)

# Gera explicações SHAP para o exame
explainer = shap.Explainer(model, X_df)
shap_values = explainer(X_df)

# Cria diretório de saída se necessário
os.makedirs("outputs/reports", exist_ok=True)

# Gera e salva o force plot (explicação individual) em HTML
html_path = "outputs/reports/shap_force_exame_benigno.html"
shap.save_html(html_path, shap.plots.force(shap_values[0]))
print(f"✅ SHAP force plot salvo em: {html_path}")