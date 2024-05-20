import pandas as pd
from pickle import load

# Carregar os modelos e normalizadores previamente treinados
crop_kmeans_model = load(open("clusterizador/crop_cluster.pkl", "rb"))
normalizador = load(open("clusterizador/normalizador.pkl", "rb"))

# Configurar a visualização de todas as colunas
pd.set_option('display.max_columns', None)

# Lista de teste
teste_instancia = [90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362]

# Criar DataFrame a partir da lista de teste
df_teste = pd.DataFrame([teste_instancia], columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Normalizar os dados da nova instância
dados_numericos_normalizados = normalizador.transform(df_teste)
dados_numericos_normalizados_df = pd.DataFrame(data=dados_numericos_normalizados, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Predizer o grupo da nova instância e a centroide correspondente
grupo_predito = crop_kmeans_model.predict(dados_numericos_normalizados_df)
centroide = crop_kmeans_model.cluster_centers_[grupo_predito]

print(f"Índice do grupo da nova instância: {grupo_predito[0]}")
print(f"Centroide da nova instância: {centroide}")

# Desnormalizar dados numéricos para legibilidade
dados_normalizados_final_legiveis = normalizador.inverse_transform(dados_numericos_normalizados)
dados_normalizados_final_legiveis_df = pd.DataFrame(data=dados_normalizados_final_legiveis, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Exibir o DataFrame final com os dados legíveis
print("Dados da nova instância (desnormalizados):")
print(dados_normalizados_final_legiveis_df.to_string(index=False))

print("Dados da nova instância (normalizados):")
print(dados_numericos_normalizados_df.to_string(index=False))
