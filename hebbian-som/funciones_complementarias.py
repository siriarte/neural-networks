# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:59:47 2017

@author: okus
"""

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def centrar(df):
    df_copia = df.copy()
    prom_col =  df_copia.mean()
    # A cada columna tengo que centrarla, restandole prom_col
    df_copia = df_copia - prom_col 
    return df_copia
    

def grafico_3d(prediccion,columna_clases,ang_elev =  None ,ang_azim = None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    longitud_pred = len(prediccion)
    xs = []
    for i in range(0,longitud_pred):
        xs.append(prediccion[i][0][0])
    ys = []
    for i in range(0,longitud_pred):
        ys.append(prediccion[i][0][1])
    zs = []
    for i in range(0,longitud_pred):
        zs.append(prediccion[i][0][2])
    ax.view_init(elev=ang_elev, azim=ang_azim)
    return ax.scatter(xs, ys, zs,c = columna_clases)
#==============================================================================
#  
# Signature: ax.view_init(elev=None, azim=None)
# Docstring:
# Set the elevation and azimuth of the axes.
# 
# This can be used to rotate the axes programatically.
# 
# 'elev' stores the elevation angle in the z plane.
# 'azim' stores the azimuth angle in the x,y plane
#==============================================================================
  