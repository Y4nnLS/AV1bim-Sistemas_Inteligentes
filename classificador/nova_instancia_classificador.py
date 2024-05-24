nova_instancia = [[79,47,39,23.689332,82.272822,6.425471,236.181114]] # Rice
# nova_instancia = [[101,28,29,25.540477,58.869846,6.790308,158.066295]] # Coffe

# Normalizar a nova instância
# Abrir o modelo normalizador salvo antes do treinamento
from pickle import load
normalizador = load(open('classificador/crop_normalizador.pwl', 'rb'))
nova_instancia_normalizada = normalizador.transform(nova_instancia)

# Classificar a nova instância
# Abrir o modelo classificador salvo anteriormente
crop_classificador = load(open('classificador/crop_tree_model_cross.pwl', 'rb'))

# Classificar
resultado = crop_classificador.predict(nova_instancia_normalizada)

dist_proba = crop_classificador.predict_proba(nova_instancia_normalizada)

# print("Classe: " + resultado)
# print(dist_proba)
import numpy as np
indice = np.argmax(dist_proba[0])
classe_predita = crop_classificador.classes_[indice]
score = dist_proba[0][indice]
print("Classificado como: ", classe_predita, "\nScore: ", str(score))
print(np.argmax(dist_proba[0]))
print(crop_classificador.classes_)