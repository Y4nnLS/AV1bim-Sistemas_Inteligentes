nova_instancia = [[37,64,22,17.48189735,18.8251973,5.954665349,121.9401369]]

# Normalizar a nova instância
# Abrir o modelo normalizador salvo antes do treinamento
from pickle import load
normalizador = load(open('C:/Users/yann_/OneDrive/Documentos/GitHub/AV1bim-Sistemas_Inteligentes/classificador/crop_normalizador.pwl', 'rb'))
nova_instancia_normalizada = normalizador.transform(nova_instancia)

# Classificar a nova instância
# Abrir o modelo classificador salvo anteriormente
crop_classificador = load(open('C:/Users/yann_/OneDrive/Documentos/GitHub/AV1bim-Sistemas_Inteligentes/classificador/crop_tree_model_cross.pwl', 'rb'))

# Classificar
resultado = crop_classificador.predict(nova_instancia_normalizada)

dist_proba = crop_classificador.predict_proba(nova_instancia_normalizada)

# print("Classe: " + resultado)
# print(dist_proba)
import numpy as np
indice = np.argmax(dist_proba[0])
classe_predita = crop_classificador.classes_[indice]
score = dist_proba[0][indice]
print("Classificado como: ", classe_predita, "Score: ", str(score))
print(np.argmax(dist_proba[0]))
print(crop_classificador.classes_)