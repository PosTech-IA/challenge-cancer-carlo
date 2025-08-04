# optimize.py
# Funções para otimizar modelos de classificação usando busca em grade (GridSearchCV)
# Explicações detalhadas para cada função
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
import numpy as np
from xgboost import XGBClassifier

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

def optimize_random_forest(X, y):
    """
    Otimiza os hiperparâmetros da Regressão Logística usando GridSearchCV.
    Testa diferentes valores de regularização e solvers.
    Retorna o melhor modelo encontrado.
    """
    print("📈 Otimizando Random Forest com GridSearchCV...")

    params = {
        'n_estimators': [50, 100, 200], # Número de árvores
        'max_depth': [None, 10, 20],   # Profundidade máxima da árvore
        'min_samples_split': [2, 5, 10]  # Mínimo de amostras para dividir um nó
    }

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=params,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )

    grid.fit(X, y)
    print("✔️  Melhor random forest encontrada:", grid.best_params_)
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

def optimize_xgboost(X, y):
    """
    Otimiza os hiperparâmetros do XGBoost usando GridSearchCV.
    Testa diferentes combinações de taxa de aprendizado e profundidade máxima.
    Retorna o melhor modelo encontrado.
    """
    print("🚀 Otimizando XGBoost com GridSearchCV...")

    params = {
        'n_estimators': [50, 100, 200], # Número de árvores
        'learning_rate': [0.01, 0.1, 0.2], # Taxa de aprendizado
        'max_depth': [3, 5, 7] # Profundidade máxima da árvore
    }

    grid = GridSearchCV(
        XGBClassifier(eval_metric='logloss', random_state=42),
        param_grid=params,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )

    grid.fit(X, y)
    print("✔️  Melhor XGBoost encontrado:", grid.best_params_)
    return grid.best_estimator_

def optimize_knn(X, y):
    """
    Otimiza os hiperparâmetros do K-Nearest Neighbors (KNN) usando GridSearchCV.
    Testa diferentes valores para o número de vizinhos.
    Retorna o melhor modelo encontrado.
    """
    print("🏃 Otimizando KNN com GridSearchCV...")

    # Como o KNN é sensível à escala, é ideal usar um pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    params = {
        'knn__n_neighbors': [3, 5, 7, 9], # Número de vizinhos
        'knn__weights': ['uniform', 'distance'] # Peso dos vizinhos
    }

    grid = GridSearchCV(
        pipe,
        param_grid=params,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )

    grid.fit(X, y)
    print("✔️  Melhor KNN encontrado:", grid.best_params_)
    return grid.best_estimator_

def optimize_svm(X, y):
    """
    Otimiza os hiperparâmetros do Support Vector Machine (SVM) usando GridSearchCV.
    Testa diferentes kernels e valores de regularização (C).
    Retorna o melhor modelo encontrado.
    """
    print("🛡️ Otimizando SVM com GridSearchCV...")
    
    # O SVM também é sensível à escala, usamos um pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42))
    ])

    params = {
        'svm__C': [0.1, 1, 10], # Parâmetro de regularização
        'svm__kernel': ['linear', 'rbf', 'poly'] # Tipos de kernel
    }

    grid = GridSearchCV(
        pipe,
        param_grid=params,
        scoring='f1',
        cv=5,
        n_jobs=-1
    )

    grid.fit(X, y)
    print("✔️  Melhor SVM encontrado:", grid.best_params_)
    return grid.best_estimator_

def evaluate_model_with_cv(model, X, y, scoring='f1'):
    """
    Avalia um modelo usando validação cruzada (cross-validation) e imprime a média e desvio padrão da métrica escolhida.
    scoring: métrica a ser usada (padrão: f1)
    """
    print(f"\n🪪 Avaliando com cross-validation (scoring = {scoring})...")
    scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
    print(f"✔️  {scoring}-score médio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")