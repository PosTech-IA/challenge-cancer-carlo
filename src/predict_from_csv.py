# predict_from_csv.py
# Faz predição do diagnóstico de câncer de mama a partir de um novo exame em CSV
# Explicações detalhadas para cada etapa

import pandas as pd
import joblib
from preprocessing import preprocess_data

# Caminho do CSV com os dados do novo exame (deve ter as mesmas colunas do treino, exceto 'diagnosis')
novo_csv = "data/exame_novo.csv"  # Arquivo de entrada

# Caminho do modelo salvo (árvore de decisão otimizada)
modelo_path = "outputs/models/tree_optimized.pkl"

# Carrega o modelo treinado
model = joblib.load(modelo_path)

# Lê o exame do arquivo CSV
df = pd.read_csv(novo_csv)

# Adiciona coluna fictícia de diagnóstico só para reutilizar o pipeline de pré-processamento
# (o valor não será usado, mas é necessário para a função preprocess_data)
df.insert(0, "diagnosis", "B")

# Pré-processa os dados do exame
X, _ = preprocess_data(df)

# Faz a predição usando o modelo carregado
prob = model.predict_proba(X)[0][1]  # Probabilidade de ser maligno
classe = model.predict(X)[0]  # Classe prevista (0=benigno, 1=maligno)

# Exibe o resultado de forma amigável
diagnostico = "Maligno" if classe == 1 else "Benigno"
print(f"🧺 Diagnóstico previsto: {diagnostico}")
print(f"📊 Probabilidade de malignidade: {prob*100:.2f}%")