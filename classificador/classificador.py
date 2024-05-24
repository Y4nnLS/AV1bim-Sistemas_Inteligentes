# CLASSIFICADOR
# Pipeline (Transformações necessárias sobre os dados)
# 1. Normalizar
# 2. Balancear
# 3. Determinar os hiperparâmetros (depois)
# 4. Treinar o modelo
# 5. Avaliar a acurácia do modelo

import warnings
warnings.filterwarnings("ignore")

# Abrir os dados
import pandas as pd
dados = pd.read_csv('./Crop_Recommendation.csv', sep=',')

# Segmentar os dados em atributos e classes
dados_atributos = dados.drop(columns=['Crop'])
dados_classes = dados['Crop']

# print(dados_atributos)
# print(dados_classes)

# 1 - Normalizar os dados
# 1.1 - Segmentar dados numéricos e dados categóricos
from sklearn.preprocessing import MinMaxScaler # Classe normalizadora

# 1.2 - Gerar o modelo normalizador para uso posterior
normalizador = MinMaxScaler()
crop_normalizador = normalizador.fit(dados_atributos) # o método fit()gera o modelo para normalização
## Salvar o modelo normalizador para uso posterior
from pickle import dump
dump(crop_normalizador, open('classificador/crop_normalizador.pwl', 'wb'))

####################################################################################################################################
# 1.3 - Normalizar a base de dados para treinamento
dados_atributos_normalizados = normalizador.fit_transform(dados_atributos) # o método fit_transform gera os dados normalizados

# 1.4 - Recompor os dados na forma de data frames
# Incorporar os dados normalizados em um único objeto
# Converter o ndarray em dataframe
dados_finais = pd.DataFrame(dados_atributos_normalizados, columns=['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH_Value','Rainfall'])
dados_finais = dados_finais.join(dados_classes, how='left')

# print(dados_finais)

####################################################################################################################################
# 2. Balancear
# Frequencia de classes conforme os dados originais
print('Freq das classes original: ', dados_classes.value_counts())

# Aplicar SMOTE (técnica que amplia a frequência das classes sintetizando novas instâncias e respeitando a probabilidade do segmento original de dados)
from imblearn.over_sampling import SMOTE
# Segmentar os dados em atributos e classes
dados_atributos = dados_finais.drop(columns=['Crop'])
dados_classes = dados_finais['Crop']

# Construir um objeto a partir do SMOTE
resampler = SMOTE()
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)

# Verificar a frequência das classes após o balanceamento
print('Frequência de classes após balanceamento')
from collections import Counter
classes_count = Counter(dados_classes_b)
print(classes_count)

# Converter os dados balanceados em DataFrames
dados_atributos_b = pd.DataFrame(dados_atributos_b)
dados_classes_b = pd.DataFrame(dados_classes_b)

print(dados_atributos_b)

####################################################################################################################################
# grid search
import numpy as np
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()

# Montagem da grade de parâmentros
# Número de árvores na floresta
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 3)]
# Número de atributos considerados em cada segmento
max_features = ['sqrt', 'log2']
# Número máximo de folhas em cada árvore
max_depth = [int(x) for x in np.linspace(10,110, num = 3)]
max_depth.append(None)
# Número mínimo de instâncias requeridas para segmentar cada nó
min_samples_split = [2,5,10]
# Número mínimo de amostras necessárias em cada nó
min_samples_leaf = [1,2,4]
# Método de seleção de amostras para treinar cada árvore
# bootstrap = [True, False]
from sklearn.model_selection import GridSearchCV
random_grid = {'max_features' : max_features,
               'max_depth' : max_depth,
               'min_samples_split' : min_samples_split,
               'min_samples_leaf' : min_samples_leaf,}

# Alternative
# random_grid = {'n_estimators' : n_estimators,
#                'max_features' : max_features,
#                'max_depth' : max_depth}

from pprint import pprint
pprint(random_grid)

# INICIAR A BUSCA PELO MELHORES HIPERPARAMETROS

# rf = instanciação da randomForest
rf_grid = GridSearchCV(tree,random_grid,refit=True,verbose=2)
rf_grid.fit(dados_atributos_b, dados_classes_b)

print("##### MELHORES HIPERPARÂMETROS #####")
print(type(rf_grid))
print(rf_grid.best_params_)
# Obter os melhores parâmetros
best_params = rf_grid.best_params_
####################################################################################################################################
# 3. Treinar

# 3.1 - Segmentar os dados em conjunto para treinamento e conjunto para testes (Test HoldOut)
from sklearn.model_selection import train_test_split
atributos_train, atributos_test, classes_train, classes_test = train_test_split(dados_atributos_b, dados_classes_b)

#3.2 - Treinar o modelo
# Será usada uma árvore, mas que pode ser substituida por outro indutor
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(**best_params)

# TREINAMENTO E TESTES COM HOLD OUT (70% treino e 30% testes)
crop_tree = tree.fit(atributos_train, classes_train) # Este treinamento não faz sentido quando usamos CrossValidation

# # pretestar o modelo
Classe_test_predict = crop_tree.predict(atributos_test)

# # Comparar as classes inferidas no teste com as classes preservadas no split
# i = 0
# for i in range(0, len(classes_test)):
#     print(classes_test.iloc[i][0], ' - ', Classe_test_predict[i])

####################################################################################################################################
# AVALIAÇÃO DA ACURÁCIA COM CROSS VALIDATION
# 3. Avaliar a acurácia com Cross-Validation
from sklearn.model_selection import cross_validate, cross_val_score
scoring = ['precision_macro', 'recall_macro']
scores_cross = cross_validate(tree, dados_atributos_b, dados_classes_b, cv = 10, scoring = scoring)

# print(scores_cross)
print(scores_cross['test_precision_macro'].mean())
print(scores_cross['test_recall_macro'].mean())
score_cross_val = cross_val_score(tree, dados_atributos_b, dados_classes_b, cv = 10)
print(score_cross_val.mean(), ' - ', score_cross_val.std())

# APÓS AVALIAR A ACURÁCIA COM O CROSS VALIDATION
# Treinar o modelo com a base normalizada, balanceada e completa
crop_tree = tree.fit(dados_atributos_b, dados_classes_b)

# Salvar o modelo para uso posterior
dump(crop_tree, open('classificador/crop_tree_model_cross.pwl', 'wb'))

####################################################################################################################################
# Acurácia global do modelo
from sklearn import metrics
print('Acurácia global (provisória): ', metrics.accuracy_score(classes_test, Classe_test_predict))

# MATRIZ DE CONTINGÊNCIA
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Redesenha a matriz de confusão no novo tamanho
ConfusionMatrixDisplay.from_estimator(crop_tree, atributos_test, classes_test)

# Ajusta a rotação dos rótulos das classes no eixo x para melhor legibilidade
plt.xticks(rotation=90, ha='right')

# Exibe a matriz de confusão
plt.show()

####################################################################################################################################
crop_tree_cross = tree.fit(dados_atributos_b, dados_classes_b)
ConfusionMatrixDisplay.from_estimator(crop_tree_cross, dados_atributos_b, dados_classes_b)
plt.xticks(rotation=90, ha='right')
plt.show()

dump(crop_tree_cross, open('classificador/crop_tree_cross.pkl', 'wb'))