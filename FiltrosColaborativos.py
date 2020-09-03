# -*- coding: utf-8 -*-

"""
Filtros Colaborativos para TUL
Aproximación de recomendación para grupos de clientes con
preferencias similares
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
import math

from datetime import datetime, date, timedelta

# =============================================================================
# Lectura de datos de entrada: Main, Clientes, Productos
# =============================================================================

data=pd.read_csv("MainData.csv")
data.columns = ['order_id', 'mes','cliente','producto','cantidad']
data['mes']=[datetime.strptime(k, '%Y-%m-%d %H:%M:%S') for k in data['mes']]

ActivClients=len(set(data['cliente']))
ActivProduct=len(set(data['producto']))

data=np.array(data)

clientes=pd.read_csv("Clientes.csv")
clientes.columns = ["cliente_id",'name','last_name']
x=[int(k) for k in clientes['cliente_id']]

productos=pd.read_csv("Productos.csv")
productos.columns = ['producto_id','name']
y=[int(k) for k in productos['producto_id']]

# =============================================================================
# Se establece el horizonte temporal de evaluación: Últimas 16 semanas
# =============================================================================

Today=datetime.now().date()
Ago1=Today-timedelta(days=28*4)
Ago2=Today-timedelta(days=28*3)
Ago3=Today-timedelta(days=28*2)
Ago4=Today-timedelta(days=28*1)

ventas=np.zeros((len(clientes),len(productos),4))

for i in range(len(data)):
    try:
        data[i,2]=x.index(data[i,2])
        data[i,3]=y.index(data[i,3])
        if data[i,1]>=Ago1 and data[i,1]<Ago2:
            data[i,1]=0
        elif data[i,1]>=Ago2 and data[i,1]<Ago3:
            data[i,1]=1
        elif  data[i,1]>=Ago3 and data[i,1]<Ago4:
            data[i,1]=2
        elif data[i,1]>=Ago4 and data[i,1]<=Today:
            data[i,1]=3
        else:
            data[i,1]=-1
    except:
        data[i,2]=-1
        data[i,1]=-1
    
    #Se llena la matriz de ventas
    if data[i,1]!=-1:
        ventas[data[i,2],data[i,3],data[i,1]]+=data[i,4]
    
print("Las dimensiones de la matriz de ventas son :",ventas.shape)

# =============================================================================
# Se calcula el Sparcity de la matriz
# =============================================================================

Spark=ventas[:,:,0]+ventas[:,:,1]+ventas[:,:,2]+ventas[:,:,3]
Sparcity=np.count_nonzero(Spark)/(len(x)*len(y))
print("El Sparcity es de:", "%.4f" % (Sparcity*100), "%")

# =============================================================================
# Estimación de los Rating
# =============================================================================

xp = len(x)
yp = len(y)

rating=np.zeros((xp,yp))
formalizar=(ventas>0)*1

for i in range(xp):
    # Se establece como la primera compra el mes -1
    for j in range(yp):
        Fst=-1
        for k in range(4):
            
            # Se encuentra el primer mes en el que compro el producto
            if (ventas[i,j,3-k]>0):
                Fst=3-k
    
        #Se llena la matriz de ratings de acuerdo a la Ec.Diseñada
        if Fst != -1:
            recompra=(formalizar[i,j,0]+formalizar[i,j,1]+formalizar[i,j,2]+formalizar[i,j,3]-1)/(4-Fst)      
            valor=sum(ventas[i,j,t]*t for t in range(Fst, 4))/sum(t for t in range(Fst,4))
            rating[i,j]=round(recompra*valor + 1,ndigits=0)

# =============================================================================
# Estimación de la matriz de similitud
# =============================================================================

similitud=np.zeros((xp,xp))

def cosine_similarity(v1,v2):
       "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
       sumxx, sumxy, sumyy = 0, 0, 0
       for i in range(len(v1)):
           x = v1[i]; y = v2[i]
           sumxx += x*x
           sumyy += y*y
           sumxy += x*y
       return sumxy/math.sqrt(sumxx*sumyy)

for i in range(xp):
    for j in range(xp):
        a=rating[i,:]
        b=rating[j,:]
        
        if (np.sum(a)>0 and np.sum(b)>0):
            similitud[i,j]=cosine_similarity(a,b)
            
# =============================================================================
# Generación de una recomendación
# =============================================================================

recomend=similitud.dot(rating)
consol=np.empty((xp+1,yp+1),dtype=object)
score=np.empty((xp+1,yp+1),dtype=object)

consol[0,0]='ID'
score[0,0]='ID'

for j in range(yp):
    consol[0,j+1]='ProdN3_'+str(j)
    score[0,j+1]='ProdN3_'+str(j)

for i in range(xp):
    
    #Se escribe el ID del cliente
    consol[i+1,0]=x[i]
    score[i+1,0]=x[i]
    
    for j in range(yp):
        consol[i+1,j+1]=productos.loc[np.argsort(-recomend[i,:])[j],'name']
        score[i+1,j+1]=recomend[i, np.argsort(-recomend[i,:])[j]]   


np.savetxt('Recomendacion.csv',consol,delimiter=";",fmt='%s')
np.savetxt('Score.csv',score,delimiter=";",fmt='%s')



