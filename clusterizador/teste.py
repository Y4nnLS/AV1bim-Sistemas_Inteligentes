import pandas as pd
from pickle import load
from sklearn.cluster import KMeans

# Carregar modelos e normalizador previamente treinados
normalizador = load(open("clusterizador/normalizador.pkl", "rb"))
crop_kmeans_model = load(open("clusterizador/crop_cluster.pkl", "rb"))

# Listas de teste
test_instances = [
    [90, 42, 43, 20.87974371, 82.00274423, 6.502985292, 202.9355362],
    [19, 51, 25, 26.80474415, 48.23991436, 3.5253661, 43.87801983],
    [39, 58, 85, 17.88776475, 15.40589717, 5.996932037, 68.54932919],
    [40, 64, 16, 16.43340342, 24.24045875, 5.926676985, 140.3717815],
    [33, 73, 23, 29.23540524, 59.38967583, 5.985792703, 103.3301803],
    [111, 79, 53, 28.31193338, 75.77363772, 6.165001278, 119.695765],
    [112, 25, 51, 25.04746944, 85.5667282, 6.932537231, 56.72496677]
]

# Criar DataFrame das novas instâncias
df_teste = pd.DataFrame(test_instances, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Normalizar as novas instâncias usando o mesmo normalizador
dados_numericos_normalizados = normalizador.transform(df_teste)

# Prever os grupos das novas instâncias
grupos_preditos = crop_kmeans_model.predict(dados_numericos_normalizados)

# Mostrar resultados
print("Grupos preditos para as novas instâncias:")
print(type(grupos_preditos))
for i, grupo_predito in enumerate(grupos_preditos):
    centroide = crop_kmeans_model.cluster_centers_[grupo_predito]
    print(f"Índice do grupo da nova instância {i+1}: {grupo_predito}")
    print(f"Centroide da nova instância {i+1}: {centroide}")


# Desnormalizar as novas instâncias para legibilidade, se necessário
dados_normalizados_final_legiveis = normalizador.inverse_transform(dados_numericos_normalizados)
dados_normalizados_final_legiveis_df = pd.DataFrame(data=dados_normalizados_final_legiveis, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Exibir o DataFrame final com os dados legíveis
print("\nDados das novas instâncias (desnormalizados):")
print(dados_normalizados_final_legiveis_df.to_string(index=False))
