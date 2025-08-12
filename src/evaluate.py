import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.base import is_classifier
from sklearn.model_selection import cross_val_score


def evaluate_models(models, X_test, y_test, feature_names):
    if hasattr(feature_names, "columns"):
        feature_names = feature_names.columns.tolist()

    for name, model in models.items():
        print(f"\n🔍 Avaliando modelo: {name}")

        if not hasattr(model, "predict"):
            print(f"❌ O modelo {name} não possui método 'predict'. Pulando.")
            continue

        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"❌ Erro ao realizar predição com o modelo {name}: {e}")
            continue

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        # AUC só se predict_proba existir
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)
            except Exception:
                roc_auc = None
        else:
            roc_auc = None

        print(f"✔️ Accuracy      : {acc:.4f}")
        print(f"✔️ Recall        : {recall:.4f}")
        print(f"✔️ F1-Score      : {f1:.4f}")
        print(f"✔️ Precision     : {precision:.4f}")
        if roc_auc is not None:
            print(f"✔️ ROC AUC       : {roc_auc:.4f}")

        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))

        # Validação cruzada (5 folds, F1)
        if is_classifier(model):
            try:
                cv_mean, cv_std = cross_validation_summary(model, X_test, y_test, cv=5)
                print(f"📊 Validação Cruzada (F1, 5 folds): Média = {cv_mean:.4f}, Std = {cv_std:.4f}")
            except Exception as e:
                print(f"⚠️ Falha na validação cruzada: {e}")

        # Salva métricas em JSON
        os.makedirs("outputs/metrics", exist_ok=True)
        metrics_path = f"outputs/metrics/metrics_{name.replace(' ', '_')}.json"
        metrics_data = {
            "model": name,
            "accuracy": acc,
            "recall": recall,
            "f1_score": f1,
            "precision": precision,
            "roc_auc": roc_auc,
        }
        if is_classifier(model):
            metrics_data["cv_f1_mean"] = cv_mean if 'cv_mean' in locals() else None
            metrics_data["cv_f1_std"] = cv_std if 'cv_std' in locals() else None

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)
        print(f"📁 Métricas salvas em: {metrics_path}")

        # Plots
        plot_confusion_matrix(y_test, y_pred, title=f'Matriz de Confusão - {name}')
        plot_feature_importance(model, feature_names, model_name=name)
        plot_roc_curve(model, X_test, y_test, model_name=name)
        explain_with_shap(model, X_test, feature_names, model_name=name)


def cross_validation_summary(model, X, y, cv=5):
    if not is_classifier(model):
        return None, None

    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    return scores.mean(), scores.std()


def plot_confusion_matrix(y_true, y_pred, title='Matriz de Confusão'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benigno", "Maligno"], yticklabels=["Benigno", "Maligno"])
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(title)
    plt.tight_layout()
    os.makedirs('outputs/reports', exist_ok=True)
    filename = f'outputs/reports/confusion_matrix_{title.replace(" ", "_")}.png'
    plt.savefig(filename)
    print(f"📁 Matriz de confusão salva em: {filename}")
    plt.close()


def plot_feature_importance(model, feature_names, model_name="Modelo"):
    if not hasattr(model, "feature_importances_") or model.feature_importances_ is None:
        print(f"⚠️  O modelo {model_name} não possui 'feature_importances_' utilizável")
        return

    importances = model.feature_importances_
    indices = importances.argsort()[::-1]

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
    plt.close()


def plot_roc_curve(model, X_test, y_test, model_name="Modelo"):
    if not hasattr(model, "predict_proba"):
        print(f"⚠️ Modelo {model_name} não suporta predict_proba. Pulando ROC.")
        return

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()

        os.makedirs('outputs/reports', exist_ok=True)
        filename = f'outputs/reports/roc_curve_{model_name.replace(" ", "_")}.png'
        plt.savefig(filename)
        print(f"📁 Curva ROC salva em: {filename}")
        plt.close()
    except Exception as e:
        print(f"⚠️ Falha ao gerar curva ROC para {model_name}: {e}")


def choose_shap_explainer(model, X_test):
    """
    Retorna o SHAP explainer adequado ao tipo de modelo.
    """
    try:
        if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
            return shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            return shap.LinearExplainer(model, X_test)
        else:
            return shap.KernelExplainer(model.predict, shap.kmeans(X_test, 10))
    except Exception as e:
        print(f"❌ Falha ao identificar explainer: {e}")
        return None


def explain_with_shap(model, X_test, feature_names, model_name="Modelo"):
    import pandas as pd
    output_dir = f'outputs/reports/shap/{model_name.replace(" ", "_")}'
    os.makedirs(output_dir, exist_ok=True)

    print("🔎 Escolhendo tipo de SHAP explainer...")

    try:
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=feature_names)
    except Exception as e:
        print(f"⚠️ Não foi possível garantir DataFrame com feature names: {e}")

    explainer = choose_shap_explainer(model, X_test)
    if explainer is None:
        print(f"⚠️ Não foi possível inicializar o explainer SHAP para {model_name}.")
        return

    try:
        print("🧠 Gerando valores SHAP...")
        shap_values = explainer(X_test)
    except Exception as e:
        print(f"❌ Erro ao gerar valores SHAP: {e}")
        return

    # Summary Plot
    print("📊 Summary Plot...")
    try:
        plt.figure(figsize=(14, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(f"{output_dir}/summary_plot.png", bbox_inches='tight')
        plt.close()
        print(f"📁 Summary Plot salvo em: {output_dir}/summary_plot.png")
    except Exception as e:
        print(f"⚠️ Erro no Summary Plot: {e}")

    # Beeswarm Plot
    print("📊 Beeswarm Plot...")
    try:
        plt.figure(figsize=(14, 8))
        shap.plots.beeswarm(shap_values, show=False)
        plt.savefig(f"{output_dir}/beeswarm_plot.png", bbox_inches='tight')
        plt.close()
        print(f"📁 Beeswarm Plot salvo em: {output_dir}/beeswarm_plot.png")
    except Exception as e:
        print(f"⚠️ Erro no Beeswarm Plot: {e}")

    # Dependence Plots
    print("📊 Dependence Plots para top 5 features...")
    try:
        feature_order = np.argsort(np.sum(np.abs(shap_values.values), axis=0))[::-1]
        top_5_features = [feature_names[i] for i in feature_order[:5]]

        for feature in top_5_features:
            print(f"  - Gerando para: {feature}")
            shap.dependence_plot(
                ind=feature,
                shap_values=shap_values.values,
                features=X_test,
                feature_names=feature_names,
                show=False
            )
            plt.savefig(f"{output_dir}/dependence_plot_{feature}.png", bbox_inches='tight')
            plt.close()
        print(f"📁 Dependence Plots salvos em: {output_dir}")
    except Exception as e:
        print(f"⚠️ Erro nos Dependence Plots: {e}")

    # Waterfall Plot (primeira instância)
    print("📊 Waterfall Plot...")
    try:
        instance_idx = 0
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[instance_idx], show=False)
        plt.savefig(f"{output_dir}/waterfall_plot_instance_{instance_idx}.png", bbox_inches='tight')
        plt.close()
        print(f"📁 Waterfall Plot salvo em: {output_dir}/waterfall_plot_instance_{instance_idx}.png")
    except Exception as e:
        print(f"⚠️ Erro no Waterfall Plot: {e}")

    # Force Plot (PNG) - apenas top 5 features do instance_idx
    print("📊 Force Plot (imagem) - top 5 features...")
    try:
        instance_idx = 0
        shap_values_instance = shap_values[instance_idx]

        # Seleciona índices das 5 features com maior valor absoluto SHAP
        top5_indices = np.argsort(np.abs(shap_values_instance.values))[-5:]

        # Criar novo objeto SHAP apenas com top5 (função interna)
        # Aqui fazemos um workaround porque shap.force_plot não aceita filtrar valores diretamente:
        # A solução simples: usa shap_values com zeros para outras features.
        import copy
        shap_values_top5 = copy.deepcopy(shap_values_instance)
        mask = np.ones_like(shap_values_instance.values, dtype=bool)
        mask[top5_indices] = False
        shap_values_top5.values[mask] = 0  # Zera as outras features

        plt.figure(figsize=(14, 4))
        shap.plots.force(shap_values_top5, matplotlib=True, show=False)
        plt.savefig(f"{output_dir}/force_plot_top5_instance_{instance_idx}.png", bbox_inches='tight')
        plt.close()
        print(f"📁 Force Plot (PNG) salvo em: {output_dir}/force_plot_top5_instance_{instance_idx}.png")
    except Exception as e:
        print(f"⚠️ Erro ao gerar Force Plot em PNG: {e}")

    # Force Plot interativo (HTML) completo
    try:
        filename_html = f"{output_dir}/force_plot_instance_{instance_idx}.html"
        shap.save_html(filename_html, shap.force_plot(shap_values[instance_idx]))
        print(f"📁 Force Plot (HTML) salvo em: {filename_html}")
        print("💡 Abra o HTML em navegador com JavaScript para visualização interativa.")
    except Exception as e:
        print(f"⚠️ Erro ao salvar Force Plot em HTML: {e}")
