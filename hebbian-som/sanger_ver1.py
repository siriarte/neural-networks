# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 11:07:23 2017

@author: okus
"""

from percepns import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from funciones_complementarias import *


print "Importo los datos"
import pandas as pd
df1 = pd.read_csv('tp2_training_dataset.csv', header=None)

print "Centro los datos y separo la columna de clases"
df2 = df1.copy()
df2_sin_clase = df2.drop(0,1)
clases = df2[0]
#df_centrado = centrar(df2)
df_centrado = centrar(df2_sin_clase)

print "Separo los datos de entrenamiento y de validacion:"
#Separo datos de entrenamiento, de validacion y de testing. Validacion me servira para evaluar mis pruebas y con el grupo de datos testing reportare la rta final del tp en cada caso.
proporcion_entrenamiento = 0.1
cant_train = int(len(df1)*proporcion_entrenamiento)
print "Cantidad datos de entrenamiento"
print cant_train
datos_entr = df_centrado[0:cant_train]
clases_entr = clases[0:cant_train]
datos_val = df_centrado[cant_train:]
clases_val = clases[cant_train:]

#print "Cantidad datos de validacion"
#print cant_train

print "Paso los datos a lista"
trainX  = datos_entr
trainX_lista =  trainX.values.tolist()
long_input = datos_entr.shape[1]
long_output = 3


print "Creo la arquitectura"
pm = PerceptronNoSup([long_input,long_output])
print "Entreno la red con la actualizacion Sanger"
pm.train_sanger(trainX_lista) 
prediccion = pm.predict(trainX)

#%%
print "Grafico las predicciones"
grafico_3d(prediccion,clases_entr,180,180)