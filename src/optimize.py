# optimize.py
# Funções para otimizar modelos de classificação usando busca em grade (GridSearchCV)
# Explicações detalhadas para cada função

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
import numpy as np

def optimize_decision_tree(X, y):
    """
    Otimiza os hiperparâmetros da Árvore de Decisão usando GridSearchCV.
    Testa várias combinações de parâmetros para encontrar a melhor árvore.
    Retorna o melhor modelo encontrado.
    """
    print("🌳 Otimizando Árvore de Decisão com GridSearchCV...")

    params = {
        'max_depth': [3, 5, 10, None],  # Profundidade máxima da árvore
        'min_samples_split': [2, 5, 10],  # Mínimo de amostras para dividir um nó
        'criterion': ['gini', 'entropy']  # Critério de divisão
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=params,
        scoring='f1',  # Usa F1-score como métrica principal
        cv=5,  # Validação cruzada com 5 divisões
        n_jobs=-1  # Usa todos os núcleos disponíveis
    )

    grid.fit(X, y)
    print("✔️  Melhor árvore encontrada:", grid.best_params_)
    return grid.best_estimator_

def optimize_logistic_regression(X, y):
    """
    Otimiza os hiperparâmetros da Regressão Logística usando GridSearchCV.
    Testa diferentes valores de regularização e solvers.
    Retorna o melhor modelo encontrado.
    """
    print("📈 Otimizando Regressão Logística com GridSearchCV...")

    params = {
        'C': [0.01, 0.1, 1, 10],  # Parâmetro de regularização
        'penalty': ['l2'],  # Tipo de penalidade
        'solver': ['lbfgs', 'liblinear']  # Algoritmos de otimização
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid=params,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )

    grid.fit(X, y)
    print("✔️  Melhor regressão encontrada:", grid.best_params_)
    return grid.best_estimator_

def evaluate_model_with_cv(model, X, y, scoring='f1'):
    """
    Avalia um modelo usando validação cruzada (cross-validation) e imprime a média e desvio padrão da métrica escolhida.
    scoring: métrica a ser usada (padrão: f1)
    """
    print(f"\n🪪 Avaliando com cross-validation (scoring = {scoring})...")
    scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
    print(f"✔️  {scoring}-score médio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")