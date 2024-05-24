import pandas as pd
from pickle import load
from sklearn.cluster import KMeans

# Carregar modelos e normalizador previamente treinados
crop_kmeans_model = load(open("clusterizador/crop_cluster.pkl", "rb"))
normalizador = load(open("clusterizador/normalizador.pkl", "rb"))

# Listas de teste
nova_instancia = [[90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362]]

# Normalizar as novas instâncias usando o mesmo normalizador
nova_instancia_normalizada = normalizador.transform(nova_instancia)

# Prever os grupos das novas instâncias
grupo_predito = crop_kmeans_model.predict(nova_instancia_normalizada)

# Mostrar resultados
centroide = crop_kmeans_model.cluster_centers_[grupo_predito]
print(f"Índice do grupo da nova instância: {grupo_predito}")
print(f"Centroide da nova instância: {centroide}")


# Desnormalizar as novas instâncias para legibilidade, se necessário
dados_normalizados_final_legiveis = normalizador.inverse_transform(nova_instancia_normalizada)
dados_normalizados_final_legiveis_df = pd.DataFrame(data=dados_normalizados_final_legiveis, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Exibir o DataFrame final com os dados legíveis
print("\nDados das novas instâncias (desnormalizados):")
print(dados_normalizados_final_legiveis_df.to_string(index=False))

df_dados_normalizados = pd.DataFrame(data=nova_instancia_normalizada, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

print("\nDados normalizados das novas instâncias:")
print(df_dados_normalizados.to_string(index=False))

# Obter o centróide
centroide = crop_kmeans_model.cluster_centers_[grupo_predito]

# Remover a dimensão extra
centroide = centroide.reshape(-1, 7)

# Transformar o centróide em um DataFrame
df_centroide = pd.DataFrame(centroide, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Desnormalizar o centróide para legibilidade
centroide_legivel = normalizador.inverse_transform(df_centroide)

# Transformar o centróide legível em um DataFrame
df_centroide_legivel = pd.DataFrame(data=centroide_legivel, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Exibir o DataFrame do centróide legível
print("\nDados do centróide (legiveis):")
print(df_centroide_legivel.to_string(index=False))
