import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score
from preprocessing import load_and_clean_data, preprocess_data

def get_top_n_features(model, feature_names, n=10):
    """
    Retorna os nomes das N features mais importantes do modelo fornecido.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    return [feature_names[i] for i in sorted_idx[:n]]

def filter_features(X, feature_names, selected_features):
    """
    Retorna o conjunto de dados X apenas com as colunas selecionadas.
    """
    df = pd.DataFrame(X, columns=feature_names)
    return df[selected_features].values

def main():
    print("\n🔎 Treinando modelo com as top-N features")

    # Etapa 1: carregamento e pré-processamento completo
    df = load_and_clean_data()
    X_full, y = preprocess_data(df)
    feature_names = df.drop(columns=["diagnosis"]).columns

    # Etapa 2: treina modelo completo para obter importância
    full_tree = DecisionTreeClassifier(random_state=42).fit(X_full, y)
    top_features = get_top_n_features(full_tree, feature_names, n=10)
    print("✔️ Features selecionadas:", top_features)

    # Etapa 3: filtra X com as top features
    X_top = filter_features(X_full, feature_names, top_features)

    # Etapa 4: split e re-treinamento
    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Etapa 5: avaliação
    print("\n📊 Avaliação com Top-N Features")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()