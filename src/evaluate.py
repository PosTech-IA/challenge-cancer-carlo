# evaluate.py
# Funções para avaliar e interpretar modelos de classificação
# Inclui métricas, gráficos de confusão, importância de variáveis e explicações SHAP

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_models(models, X_test, y_test):
    """
    Avalia todos os modelos fornecidos usando métricas padrão (acurácia, recall, F1-score).
    Também imprime o relatório de classificação e plota a matriz de confusão.
    """
    for name, model in models.items():
        print(f"\n🔍 Avaliando modelo: {name}")

        y_pred = model.predict(X_test)  # Faz predições

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"✔️ Accuracy     : {acc:.4f}")
        print(f"✔️ Recall       : {recall:.4f}")
        print(f"✔️ F1-Score     : {f1:.4f}")

        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred, title=f'Matriz de Confusão - {name}')

def plot_confusion_matrix(y_true, y_pred, title='Matriz de Confusão'):
    """
    Plota a matriz de confusão para visualizar acertos e erros do modelo.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benigno", "Maligno"], yticklabels=["Benigno", "Maligno"])
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    os.makedirs('outputs/reports', exist_ok=True)
    filename = f'outputs/reports/confusion_matrix_{title.replace(' ', '_')}.png'
    plt.savefig(filename)
    print(f"📁 Matriz de confusão salva em: {filename}")
    plt.close()

def plot_feature_importance(model, feature_names, model_name="Modelo"):
    """
    Plota e salva a importância das variáveis (features) para modelos que possuem o atributo 'feature_importances_'.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"⚠️  O modelo {model_name} não possui 'feature_importances_'")
        return

    importances = model.feature_importances_
    indices = importances.argsort()[::-1]  # Ordena da mais importante para a menos

    plt.figure(figsize=(10, 6))
    plt.title(f"Importância das Features - {model_name}")
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.xlabel("Importância")
    plt.ylabel("Variável")
    plt.tight_layout()

    os.makedirs('outputs/reports', exist_ok=True)
    filename = f'outputs/reports/feature_importance_{model_name.replace(" ", "_")}.png'
    plt.savefig(filename)
    print(f"📁 Gráfico salvo em: {filename}")

def explain_with_shap(model, X_train, feature_names, model_name="Modelo"):
    """
    Gera explicações SHAP para o modelo, mostrando o impacto de cada variável na predição.
    Salva gráficos summary e force plot.
    """
    print(f"\n🔍 Gerando explicações SHAP para: {model_name}")
    
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # SHAP summary plot (resumo global das importâncias)
    shap.summary_plot(shap_values, features=X_train, feature_names=feature_names, show=False)
    os.makedirs('outputs/reports', exist_ok=True)
    summary_path = f'outputs/reports/shap_summary_{model_name.replace(" ", "_")}.png'
    plt.savefig(summary_path)
    print(f"📁 SHAP summary salvo em: {summary_path}")

    # SHAP force plot (explicação individual)
    force_path = f'outputs/reports/shap_force_{model_name.replace(" ", "_")}.html'
    shap.save_html(force_path, shap.plots.force(shap_values[0]))
    print(f"📁 SHAP force plot salvo em: {force_path}")