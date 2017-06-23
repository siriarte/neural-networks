import csv
import numpy as np

class hebbian(object):

    file_path_dataset = 'tp2_training_dataset.csv'
    W = []

    # Parseo el dataset
    def parsear_dataset(self):
        f = open(self.file_path_dataset, 'rt')
        data = []
        try:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        finally:
            f.close()
        return data

    # Genero pesos aleatorios
    def matriz_pesos_aleatorios(self, size):
        self.W = []
        for _ in range(size):
            self.W.append(np.random.uniform(-0.1, 0.1, size))




def init():
    h = hebbian()
    h.matriz_pesos_aleatorios(5)
    print(h.W)