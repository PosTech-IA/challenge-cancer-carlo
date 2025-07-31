import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_cancer_eda(df: pd.DataFrame, output_dir="analysis"):
    os.makedirs(output_dir, exist_ok=True)

    # Distribuição da variável alvo
    plt.figure(figsize=(6, 4))
    sns.countplot(x='diagnosis', data=df)
    plt.title("Distribuição do Diagnóstico (Benigno/Maligno)")
    plt.savefig(f"{output_dir}/diagnostico_hist.png")
    plt.close()

    # Heatmap de correlação
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=False)
    plt.title("Mapa de Correlação entre Features")
    plt.savefig(f"{output_dir}/correlacao_heatmap.png")
    plt.close()

    # Principais variáveis numéricas para análise
    num_cols = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "concavity_mean", "compactness_mean", "symmetry_mean"
    ]

    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=col, hue='diagnosis', kde=True, bins=30)
        plt.title(f"Distribuição de {col} por Diagnóstico")
        plt.savefig(f"{output_dir}/{col}_hist.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x='diagnosis', y=col)
        plt.title(f"{col} vs Diagnóstico")
        plt.savefig(f"{output_dir}/{col}_boxplot.png")
        plt.close()

if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")

    # Remove colunas não úteis
    df.drop(columns=['Unnamed: 32', 'id'], inplace=True, errors='ignore')

    run_cancer_eda(df)