# Challenge Cancer - Pipeline de Diagnóstico de Câncer de Mama

Este projeto implementa um pipeline completo de Machine Learning para análise e predição de câncer de mama a partir de dados clínicos.

## Visão Geral do Pipeline

O pipeline é composto pelas seguintes etapas principais:

1. **Carregamento e Pré-processamento dos Dados**

   - Leitura do arquivo CSV com os dados dos exames.
   - Limpeza de colunas desnecessárias.
   - Codificação do diagnóstico (benigno/maligno) para números.
   - Normalização dos dados para facilitar o aprendizado dos modelos.

2. **Divisão dos Dados**

   - Separação dos dados em conjuntos de treino e teste (normalmente 80% treino, 20% teste).

3. **Treinamento dos Modelos**

   - Treinamento de dois modelos principais: Regressão Logística e Árvore de Decisão.
   - Treinamento padrão (com parâmetros padrão) e otimizado (buscando os melhores parâmetros).

4. **Avaliação dos Modelos**

   - Avaliação dos modelos usando métricas como acurácia, recall e F1-score.
   - Geração de relatórios e matrizes de confusão para visualizar os resultados.

5. **Interpretação dos Modelos**

   - Análise da importância das variáveis (features) para a Árvore de Decisão.
   - Explicação das decisões do modelo de Regressão Logística usando SHAP (interpretação de modelos).

6. **Salvamento dos Modelos**

   - Os melhores modelos treinados são salvos para uso futuro.

7. **Predição e Explicação para Novos Exames**
   - Possibilidade de prever o diagnóstico de um novo exame a partir de um arquivo CSV.
   - Geração de explicações individuais para cada exame usando SHAP.

## Associação das Etapas do Pipeline com os Arquivos

| Etapa                                         | Arquivo(s) Responsável(is)                 |
| --------------------------------------------- | ------------------------------------------ |
| 1. Carregamento e Pré-processamento dos Dados | `preprocessing.py`                         |
| 2. Divisão dos Dados                          | `preprocessing.py`                         |
| 3. Treinamento dos Modelos                    | `train.py`                                 |
| 4. Avaliação dos Modelos                      | `evaluate.py`                              |
| 5. Interpretação dos Modelos                  | `evaluate.py`, `explain_single.py`         |
| 6. Salvamento dos Modelos                     | `train.py`, `main.py`                      |
| 7. Predição e Explicação para Novos Exames    | `predict_from_csv.py`, `explain_single.py` |

O arquivo `main.py` orquestra todo o pipeline, chamando as funções dos arquivos acima na ordem correta.
O arquivo `test_preprocessing.py` testa o pipeline de pré-processamento.

---

## Estrutura dos Arquivos (pasta `src/`)

- **main.py**: Executa todo o pipeline, do carregamento dos dados ao salvamento dos modelos e geração de relatórios.
- **preprocessing.py**: Funções para carregar, limpar, pré-processar e dividir os dados.
- **train.py**: Funções para treinar os modelos de classificação.
- **evaluate.py**: Funções para avaliar os modelos, gerar gráficos e explicações.
- **optimize.py**: Funções para otimizar os modelos usando busca de melhores parâmetros (GridSearchCV).
- **explain_single.py**: Gera explicações SHAP para um exame individual (novo paciente).
- **predict_from_csv.py**: Faz a predição do diagnóstico de um novo exame a partir de um arquivo CSV.
- **test_preprocessing.py**: Testa o pipeline de pré-processamento para garantir que está funcionando corretamente.

---

## Como Usar

### Localmente

0. Ative o venv
   `source venv/bin/activate`
1. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download do [dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)**
3. **Coloque o arquivo de dados em `data/data.csv`** (ou ajuste o caminho nos scripts).
4. **Execute o pipeline principal**:
   ```bash
   python src/main.py
   ```
5. **Para testar um novo exame**:
   - Coloque o exame em `data/exame_novo.csv` (com as mesmas colunas do treino).
   - Execute:
     ```bash
     python src/predict_from_csv.py
     ```
   - Para explicação individual:
     ```bash
     python src/explain_single.py
     ```

### Docker

1. **Coloque o arquivo de dados em `data/data.csv`** (ou ajuste o caminho nos scripts).
2. **Execute o docker-compose**:

   ```bash
   pela primeira vez rode:
   docker-compose up --build

   caso contrário:
   docker-compose up
   ```

---

## Observações

- Todos os scripts estão comentados e documentados para facilitar o entendimento.
- Os resultados e modelos são salvos na pasta `analysis/`.
- O projeto utiliza bibliotecas populares como `scikit-learn`, `pandas`, `matplotlib`, `seaborn` e `shap`.

---
