#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 12:56:22 2025
@author: safaaru

Mega Case Study: Hybrid Deep Learning Model for Fraud Detection
PART 1: Unsupervised Learning with Self-Organizing Maps (SOM)
PART 2: Supervised Learning with Artificial Neural Networks (ANN)
"""

# -----------------------------
# PART 1: Identify Frauds Using SOM
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')

# Features and target
X = dataset.iloc[:, :-1].values  # All columns except last
y = dataset.iloc[:, -1].values   # Last column (approval)

# Normalize features to [0, 1] range
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X = min_max_scaler.fit_transform(X)

# Train Self-Organizing Map
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualization of SOM
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)  # Distance map visualization
colorbar()

# Markers and colors for actual classifications
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Extract suspected frauds from SOM
mappings = som.win_map(X)

# Option 1: Hardcoded suspicious neurons (less ideal)
frauds = np.concatenate((mappings[(1,1)], mappings[(8,5)]), axis=0)

# -----------------------------
# Option 2 (Recommended): Extract all high-distance neurons automatically
#distance_map = som.distance_map().T
#threshold = 0.9  # Define a threshold for suspicious distance values
#fraud_coords = np.argwhere(distance_map > threshold)
#frauds = np.concatenate([mappings.get(tuple(coord), []) for coord in fraud_coords if tuple(coord) in mappings], axis=0)
# -----------------------------

# Reverse normalization to get actual values
frauds = min_max_scaler.inverse_transform(frauds)

# -----------------------------
# PART 2: From Unsupervised to Supervised Deep Learning
# -----------------------------

# Features for supervised model
customers = dataset.iloc[:, :-1].values

# Initialize dependent variable (fraud flag)
is_fraud = np.zeros(len(dataset))

# Set is_fraud = 1 for customer IDs detected by SOM
fraud_ids = set(frauds[:, 0])  # Assuming first column is CustomerID
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in fraud_ids:
        is_fraud[i] = 1

# Standardize features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# -----------------------------
# Building the ANN
# -----------------------------

from keras.models import Sequential
from keras.layers import Dense

# Initialize the ANN
model = Sequential()

# Input layer and first hidden layer
model.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))

# Output layer
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compile the ANN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the ANN
model.fit(customers, is_fraud, batch_size=1, epochs=6)

# -----------------------------
# PART 3: Predicting Fraud Probabilities
# -----------------------------

# Predict fraud probabilities
y_pred = model.predict(customers)

# Combine predictions with customer IDs
y_pred_combined = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)

# Sort by fraud probability (highest risk at top)
y_pred_sorted = y_pred_combined[y_pred_combined[:, 1].argsort()[::-1]]

# Optional: Display as DataFrame for clarity
result_df = pd.DataFrame(y_pred_sorted, columns=['CustomerID', 'FraudProbability'])
print(result_df.head(8))  # Show top 10 suspected frauds
