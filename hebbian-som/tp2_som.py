from som import *
import parser
import tp2 as hebbian

def prueba():
    categorias, data_set = hebbian.parsear_dataset()
    data_set_centrado = hebbian.centrar_matriz(data_set)
    red = PerceptronSOM(len(data_set_centrado[0]), 7, 7)
    red.train_som(data_set_centrado)
    red.graficar(data_set, categorias)
    

prueba()
