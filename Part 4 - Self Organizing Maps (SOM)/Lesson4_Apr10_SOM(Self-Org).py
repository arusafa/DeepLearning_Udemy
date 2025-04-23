#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 10:59:24 2025

@author: safaaru
"""

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scalling

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))

#Fitting sc to X
X = sc.fit_transform(X) #Normalization (making X between 0 and 1)

# Training the SOM

from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)

som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
#
from pylab import bone, pcolor, colorbar, plot, show

#the window that contains the map is bone
bone()

# naming nodes with different colour #To take the transpose we add .T 
pcolor(som.distance_map().T) 

# Adding lagends 
colorbar()

# Adding markers 'o' is a circle , 's' is a square
markers = ['o', 's' ]

# Coloring the markers 'r' is a red color , 'g' is a green color
colors = ['r' , 'g' ]

# Looping each customer in order to get the winning nodes and if the customer is approved green square

#i is going to be the index from 0 to 689 and x is going to be different vectors of the customers
for i, x in enumerate(X):
    #getting the winning nodes for the first customer
    w = som.winner(x)
    #placing markers on the winning nodes
    #putting marker in the center
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor =  colors[y[i]], #placing colors
         markerfacecolor =  'None', #getting inside color of the marker
         markersize = 10 ,
         markeredgewidth = 2)
show()

# Finding the frauds

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,7)]) , axis = 0)

# Inverse the scalling
frauds = sc.inverse_transform(frauds)






