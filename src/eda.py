import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def save_individual_hist_and_box(df, features, output_dir):
    """
    Salva histogramas e boxplots individuais para cada feature.
    """
    os.makedirs(output_dir, exist_ok=True)
    for col in features:
        # Histograma individual
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue='diagnosis', kde=True, bins=30)
        plt.title(f"Distribuição de {col} por Diagnóstico")
        plt.savefig(f"{output_dir}/{col}_hist.png")
        plt.close()

        # Boxplot individual
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x='diagnosis', y=col)
        plt.title(f"{col} vs Diagnóstico")
        plt.savefig(f"{output_dir}/{col}_boxplot.png")
        plt.close()


def save_pairplot(df, features, output_dir, group_name=""):
    """
    Gera um pairplot das features especificadas.
    """
    os.makedirs(output_dir, exist_ok=True)
    pairplot_features = features + ['diagnosis']
    pairplot = sns.pairplot(df[pairplot_features], hue='diagnosis', diag_kind='kde')
    pairplot.fig.suptitle(f'Pairplot das Features {group_name}', y=1.02)

    # Ajusta margens laterais e superior para evitar cortes
    pairplot.fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, hspace=0.3, wspace=0.3)

    pairplot.fig.savefig(f"{output_dir}/pairplot_{group_name.lower()}.png")
    plt.close(pairplot.fig)


def save_boxplots_grid(df, features, output_dir, group_name):
    import math
    os.makedirs(output_dir, exist_ok=True)

    n = len(features)
    cols = 3  # número de colunas no grid (ajuste se quiser)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.boxplot(data=df, x='diagnosis', y=col, ax=axes[i])
        axes[i].set_title(f"{col} vs Diagnóstico")
        axes[i].set_xlabel('Diagnóstico')
        axes[i].set_ylabel(col)

    # Remove axes vazios (se houver)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f'Boxplots Individuais - Grupo {group_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Ajusta margens para dar mais espaço lateral e superior
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)
    
    filepath = f"{output_dir}/boxplots_{group_name.lower()}.png"
    plt.savefig(filepath)
    plt.close()
    print(f"📁 Boxplots agrupados salvos em: {filepath}")

def run_cancer_eda(df: pd.DataFrame, output_dir="analysis"):
    os.makedirs(output_dir, exist_ok=True)

    # Diagnóstico
    plt.figure(figsize=(6, 4))
    sns.countplot(x='diagnosis', data=df)
    plt.title("Distribuição do Diagnóstico (Benigno/Maligno)")
    plt.savefig(f"{output_dir}/diagnostico_hist.png")
    plt.close()

    # Heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False)
    plt.title("Mapa de Correlação entre Features")
    plt.savefig(f"{output_dir}/correlacao_heatmap.png")
    plt.close()

    # Feature groups
    features_mean = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean",
        "symmetry_mean", "fractal_dimension_mean"
    ]

    features_worst = [
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
        "symmetry_worst", "fractal_dimension_worst"
    ]

    features_se = [
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
        "symmetry_se", "fractal_dimension_se"
    ]

    # Diretórios específicos para cada grupo
    mean_dir = os.path.join(output_dir, "mean")
    worst_dir = os.path.join(output_dir, "worst")
    se_dir = os.path.join(output_dir, "se")

    # Salva gráficos por grupo
    save_individual_hist_and_box(df, features_mean, mean_dir)
    save_boxplots_grid(df, features_mean, mean_dir, "Mean")
    save_pairplot(df, features_mean, mean_dir, group_name="Mean")

    save_individual_hist_and_box(df, features_worst, worst_dir)
    save_boxplots_grid(df, features_worst, worst_dir, "Worst")
    save_pairplot(df, features_worst, worst_dir, group_name="Worst")

    save_individual_hist_and_box(df, features_se, se_dir)
    save_boxplots_grid(df, features_se, se_dir, "SE")
    save_pairplot(df, features_se, se_dir, group_name="SE")

if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    df.drop(columns=['Unnamed: 32', 'id'], inplace=True, errors='ignore')
    run_cancer_eda(df)
