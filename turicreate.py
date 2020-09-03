# -*- coding: utf-8 -*-
"""
SdR TUL
"""

import pandas as pd
import numpy as np


# =============================================================================
# Pre-procesamiento de los datos
# =============================================================================

# Se leen los datos de rating de productos
df=pd.read_csv('rating.csv')

# Los datos de Usuarios e Items no están ordenados secuencialmente.
# Se agregan columnas de index para codificarlos mas fácil

# Se reescribe el ID de los items
unique_clients=set(df.usuario.values)
unique_items=set(df.item.values)

user_ix = {}
item_ix = {}

count=0  
for i in unique_clients:
    user_ix[i]=count
    count+=1

count=0
for i in unique_items:
    item_ix[i]=count
    count+=1
    
df['userIndex']=df.apply(lambda x: user_ix[x['usuario']], axis = 1)
df['itemIndex']=df.apply(lambda x: item_ix[x['item']], axis = 1)


# =============================================================================
# Turicreate
# =============================================================================

sf = tc.SFrame({'user_id': df['usuario'],
...                       'item_id': df['item']})
m = tc.item_similarity_recommender.create(sf)
nn = m.get_similar_items()
m2 = tc.item_similarity_recommender.create(sf, nearest_items=nn)
recs = m.recommend()
print(recs)