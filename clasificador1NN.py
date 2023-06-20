import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#creamos las columnas para los datos
#columnas = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']

#Cargamos los datos
datos = pd.read_csv('bezdekIris.csv')

#Separamos los datos en X y Y y creamos las matrices
dato = datos.values
X = dato[:,0:4]
Y = dato[:,4]

# Dividimos los datos en entrenamiento y prueba utilizando Leave One Out
loo = LeaveOneOut()

#calculamos que valor de k es mejor para el clasificador
ks = 10 #vamos a evaluar hasta k = 20
media_precision = np.zeros(ks) #vector que almacena la media de precision de los disntintos k

#n iterara hasta ks-1
for n in range(1, ks+1):
    precision_sum = 0.0
    #iteramos sobre cada split de entrenamiento y prueba
    for train_index, test_index in loo.split(X):
        #seleccionamos los datos correspondientes utilizando esos índices y los almacenamos en las listas correspondientes
        X_train = X[train_index]
        X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]

        # entrenamos con k
        vecino = KNeighborsClassifier(n_neighbors=n).fit(X_train, Y_train)

        # realizamos la predicción
        prediccion = vecino.predict(X_test)

        # calculamos la precision
        precision = accuracy_score(Y_test, prediccion)

        # Acumulamos los resultados de precisión
        precision_sum += precision

    # Calculamos la media de precisión y desviación estándar para el valor de k actual
    media_precision[n - 1] = precision_sum / X.shape[0]

for k, precision in zip(range(1, ks + 1), media_precision):
    print("K =", k, "  Media de Precisión =", precision)

