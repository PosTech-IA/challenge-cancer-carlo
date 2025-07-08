# train.py
# Funções para treinar modelos de classificação para câncer de mama
# Explicações detalhadas para cada função

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def train_logistic_regression(X_train, y_train):
    """
    Treina um modelo de Regressão Logística.
    X_train: dados de entrada de treino
    y_train: rótulos de treino
    Retorna o modelo treinado.
    """
    model = LogisticRegression(max_iter=1000)  # max_iter garante que o modelo irá convergir
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    """
    Treina um modelo de Árvore de Decisão.
    X_train: dados de entrada de treino
    y_train: rótulos de treino
    Retorna o modelo treinado.
    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def train_all_models(X_train, y_train):
    """
    Treina todos os modelos disponíveis e retorna um dicionário com os nomes e modelos.
    """
    models = {}

    print("🔧 Treinando Regressão Logística...")
    models['Logistic Regression'] = train_logistic_regression(X_train, y_train)

    print("🌳 Treinando Árvore de Decisão...")
    models['Decision Tree'] = train_decision_tree(X_train, y_train)

    print("✅ Modelos treinados com sucesso.")
    return models