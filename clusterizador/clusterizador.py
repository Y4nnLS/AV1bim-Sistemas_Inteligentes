import pandas as pd
from pickle import dump
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import cdist
import numpy as np

# Carregar os dados
dados = pd.read_csv('C:/Users/Usuário/Documents/GitHub/AV1bim-Sistemas_Inteligentes/Crop_Recommendation.csv', sep=',')

# Remover a coluna 'Crop'
dados_numericos = dados.drop(columns=['Crop'])

# Normalizar os dados
normalizador = preprocessing.MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_numericos)
dados_numericos_normalizados = modelo_normalizador.transform(dados_numericos)

# Converter para DataFrame
dados_numericos_normalizados = pd.DataFrame(data=dados_numericos_normalizados, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Inverter a normalização para legibilidade
dados_normalizados_final_legiveis = modelo_normalizador.inverse_transform(dados_numericos_normalizados)
dados_normalizados_final_legiveis = pd.DataFrame(data=dados_normalizados_final_legiveis, columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

# Mostrar todas as colunas no output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 2)

print("Dados normalizados legíveis:")
print(dados_normalizados_final_legiveis.to_string(index=True))
# pd.set_option('display.max_columns', None)
# print(f'Dados normalizados legíveis: {dados_normalizados_final_legiveis}')

# Salvar o normalizador em um arquivo pickle
dump(modelo_normalizador, open("clusterizador/normalizador.pkl", "wb"))

# Encontrar o número ótimo de clusters usando o método Elbow
distortions = []
K = range(1, 101)

for i in K:
    crop_kmeans_model = KMeans(n_clusters=i, n_init='auto', random_state=42).fit(dados_normalizados_final_legiveis)
    distortions.append(sum(np.min(cdist(dados_normalizados_final_legiveis, crop_kmeans_model.cluster_centers_, 'euclidean'), axis=1) / dados_normalizados_final_legiveis.shape[0]))

print(distortions)

# Plotar o gráfico Elbow
fig, ax = plt.subplots()
ax.plot(K, distortions)
ax.set(xlabel='n Clusters', ylabel='Distorção', title='Elbow pela distorção')
ax.grid()
fig.savefig('clusterizador/elbow_distorcao.png')
plt.show()

# Calcular o número ótimo de clusters
x0, y0 = K[0], distortions[0]
xn, yn = K[-1], distortions[-1]

# Iterar nos pontos gerados durante os treinamentos preliminares
distancias = []
for i in range(len(distortions)):
    x, y = K[i], distortions[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

# Maior distância
n_clusters_otimo = K[distancias.index(np.max(distancias))]

# Treinar o modelo KMeans com o número ótimo de clusters
crop_kmeans_model = KMeans(n_clusters=n_clusters_otimo, n_init='auto').fit(dados_normalizados_final_legiveis)

# Salvar o modelo treinado em um arquivo pickle
dump(crop_kmeans_model, open("clusterizador/crop_cluster.pkl", "wb"))

print(crop_kmeans_model.cluster_centers_)
