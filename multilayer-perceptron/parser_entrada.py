import csv
import numpy as np
import random as random

ejercicio_1_file = 'tp1_ej1_training.csv'
ejercicio_2_file = 'tp1_ej2_training.csv'


def normalizar_vector_ej2(vector):
    min_index = vector.argmin()
    max_index = vector.argmax()
    min = vector[min_index]
    max = vector[max_index]

    vector_normalizado = []

    for value in vector:
        vector_normalizado.append(2*(value-min)/((max-min))-1)

    v = np.array(vector_normalizado)
    min_index = v.argmin()
    max_index = v.argmax()
    min = v[min_index]
    max = v[max_index]

    return vector_normalizado

def normalizar_vector_ej1(vector):
    media = np.mean(vector)
    desvio = np.std(vector)

    vector_normalizado = []

    for value in vector:
        vector_normalizado.append((value - media) / desvio)

    return vector_normalizado


def normalizar_matriz(matriz, ejercicio):
    ret = []
    primera_columna = True
    for c in matriz:
        if(ejercicio==1):
            if primera_columna:
                ret.append(c)
                primera_columna = False
            else:
                ret.append(normalizar_vector_ej1(c))
        else:
            ret.append(normalizar_vector_ej2(c))
    return ret


def datos_ejercicio_1():

    f = open(ejercicio_1_file, 'rt')
    data = []
    try:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    finally:
        f.close()

    data_parsed = []
    for row in data:
        row_fixed = []
        if(row[0]=='M'):
            row_fixed.append(float(0))
        else:
            row_fixed.append(float(1))
        for i in range(1, len(row)):
            row_fixed.append(float(row[i]))
        data_parsed.append(row_fixed)

    data_parsed = np.array(data_parsed)
    data_parsed_transpuesta = data_parsed.transpose()
    data_normalizada_traspuesta = normalizar_matriz(data_parsed_transpuesta, 1)
    data_normalizada_traspuesta = np.array(data_normalizada_traspuesta)
    data_normalizada =  data_normalizada_traspuesta.transpose().tolist()
    random.shuffle(data_normalizada)

    datos_entrada = []
    datos_resultado = []
    for c in data_normalizada:
        datos_entrada.append(c[1:])
        datos_resultado.append([c[0]])

    return {'entrada': datos_entrada, 'resultado': datos_resultado}


def datos_ejercicio_2():

    f = open(ejercicio_2_file, 'rt')
    data = []
    try:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    finally:
        f.close()

    data_parsed = []
    for row in data:
        row_fixed = []
        for i in range(0, len(row)):
            row_fixed.append(float(row[i]))
        data_parsed.append(row_fixed)

    data_parsed = np.array(data_parsed)
    data_parsed_transpuesta = data_parsed.transpose()
    data_normalizada_traspuesta = normalizar_matriz(data_parsed_transpuesta, 2)
    data_normalizada_traspuesta = np.array(data_normalizada_traspuesta)
    data_normalizada = data_normalizada_traspuesta.transpose().tolist()
    random.shuffle(data_normalizada)

    datos_entrada = []
    datos_resultado = []
    for c in data_normalizada:
        datos_entrada.append(c[0:8])
        datos_resultado.append([c[8],c[9]])

    return {'entrada': datos_entrada, 'resultado': datos_resultado}