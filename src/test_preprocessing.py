# test_preprocessing.py
# Testa o pipeline de pré-processamento dos dados
# Explicações detalhadas para cada etapa

from preprocessing import load_and_clean_data, preprocess_data, split_data

def main():
    print("🚀 Iniciando teste de pré-processamento...")

    print("🔹 Carregando dados...")
    df = load_and_clean_data()
    print("✔️  Shape do dataframe:", df.shape)
    print("✔️  Colunas:", df.columns.tolist())

    if df.empty:
        print("❌ ERRO: O dataframe está vazio. Verifique o conteúdo do CSV.")
        return

    print("\n🔹 Pré-processando dados...")
    X, y = preprocess_data(df)
    print("✔️  Shape de X:", X.shape)
    print("✔️  Shape de y:", y.shape)

    print("\n🔹 Separando treino e teste...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("✔️  Train size:", X_train.shape[0])
    print("✔️  Test size:", X_test.shape[0])

    print("\n✅ Teste de pré-processamento finalizado com sucesso!")

if __name__ == "__main__":
    main()