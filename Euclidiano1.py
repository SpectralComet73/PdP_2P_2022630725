import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import numpy as np

class MDC():
    def __init__(self): #Constructor de la clase
        #Atributos de la clase
        self.X_train = None #Datos de entrenamiento
        self.y_train = None
        self.classes = None #Clases del dataset (Especies de Iris)
        self.class_centroids = None #Centroides de cada clase , los centroides son los promedios de cada clase
    
    def fit(self, X_train, y_train): #Entrenamiento del clasificador
        self.X_train = X_train #Se guardan los datos de entrenamiento
        self.y_train = y_train
        self.classes = np.unique(y_train)
        
        self.class_centroids = {} #Se crea un diccionario vacío para guardar los centroides de cada clase
        for class_label in self.classes: # Para cada clase se calcula su centroide
            class_data = X_train[y_train == class_label] # Se obtienen los datos de la clase
            class_centroid = np.mean(class_data, axis=0) # Se calcula el centroide de la clase
            self.class_centroids[class_label] = class_centroid # Se guarda el centroide en el diccionario
    
    def predict(self, X_test): # Predicción de las clases de los datos de prueba (X_test) 
        y_pred = [] # Lista para guardar las clases predichas
        for x in X_test: # Para cada dato de prueba se calcula la distancia a cada centroide
            distances = cdist([x], list(self.class_centroids.values())) # Distancia entre el dato de prueba y cada centroide
            nearest_class = np.argmin(distances) # Se obtiene el índice del centroide más cercano
            y_pred.append(list(self.class_centroids.keys())[nearest_class]) # Se guarda la clase predicha
        return y_pred # Se regresa la lista de clases predichas

iris = pd.read_csv('bezdekIris.data', header=None, names=['Longitud_Cepalo_Cm', 'Ancho_Cepalo_Cm', 'Longitud_Petalo_Cm', 
                                                          'Ancho_Petalo_Cm', 'Especie'])
#Se imprime la lista de datos
print(iris)
#Se extraen las caracteristicas y las etiquetas de destino siendo x las caracteristicas e y las etiquetas
X = iris.drop(columns='Especie').values
y = iris['Especie'].values

# Técnica Hold-Out 70E/30P
# Dividir el dataset en entrenamiento y prueba (70% y 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Entrenar el clasificador
mdc_holdout = MDC()
mdc_holdout.fit(X_train, y_train)

# Predecir en el conjunto de entrenamiento y prueba``
y_pred_train = mdc_holdout.predict(X_train)
y_pred_test = mdc_holdout.predict(X_test)

# Calcular el accuracy en el conjunto de entrenamiento y prueba
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Se imprime la precision en el entrenamiento y en la prueba
print("Precision en el entrenamiento: {:.2f}%".format(accuracy_train * 100))
print("Precision en la prueba: {:.2f}%".format(accuracy_test * 100))
