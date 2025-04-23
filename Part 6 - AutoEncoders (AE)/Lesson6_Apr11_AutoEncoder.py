#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 09:39:41 2025
@author: safaaru
"""

# -*- coding: utf-8 -*-

# ========================== Importing Libraries ========================== #
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# ========================== Loading the Dataset ========================== #
# MovieLens 1M metadata (Movies, Users, Ratings)
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Training and test sets from the 100k version (for simplicity)
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# ========================== User & Movie Info ========================== #
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))      # total users
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))    # total movies

# ========================== Convert Data to Matrix ========================== #
# Each row = one user
# Each column = one movie
# Cell = rating (0 if unrated)

def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# ========================== Normalize + Convert to Tensors ========================== #
# Ratings are scaled from 1–5 → 0–1 to match ReLU output better

training_set = torch.FloatTensor(np.array(training_set) / 5.0)
test_set = torch.FloatTensor(np.array(test_set) / 5.0)


# ========================== AutoEncoder Model ========================== #
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()

        # --------------------- ENCODING LAYERS --------------------- #
        # These reduce dimensionality — learning compressed latent representations (vectors of user preferences)
        self.fc1 = nn.Linear(nb_movies, 20)   # Input Layer → 20 hidden neurons
        self.fc2 = nn.Linear(20, 10)          # Latent vector layer (bottleneck, vector of preferences)

        # --------------------- DECODING LAYERS --------------------- #
        # These reconstruct the original input (user's rating vector) from compressed form
        self.fc3 = nn.Linear(10, 20)          # Expanding back to 20
        self.fc4 = nn.Linear(20, nb_movies)   # Output layer → reconstruct original movie vector

        # Activation function: Sigmoid squashes values between 0 and 1
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        # --------------------- ENCODING --------------------- #
        # Input user rating vector gets transformed into a latent vector
        x = torch.relu(self.fc1(x))   # Input → 20-dim
        x = torch.relu(self.fc2(x))   # 20 → 10-dim (latent features)

        # --------------------- DECODING --------------------- #
        # Try to reconstruct the original rating vector from latent representation
        x = torch.relu(self.fc3(x))   # 10 → 20
        x = self.fc4(x)                    # 20 → nb_movies (final prediction)
        return x

# ========================== Training Setup ========================== #
sae = SAE()                                # Model instance
criterion = nn.MSELoss()                   # Mean Squared Error loss
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.01)  # Optimizer with L2 regularization


# ========================== Training the AutoEncoder ========================== #

losses = []

nb_epoch = 200  # Number of times the model will see the entire training set
for epoch in range(1, nb_epoch + 1):
    train_loss = 0  # Accumulator for total RMSE loss
    s = 0.          # Counter for valid users (users who rated at least 1 movie)

    for id_user in range(nb_users):
        # ------------------  Get Input Vector for One User ------------------ #
        # Example: A vector of length 1682 (number of movies), with ratings as values
        input_vector = Variable(training_set[id_user]).unsqueeze(0)  # Add batch dimension: shape (1, nb_movies)
        target = input_vector.clone()  # Target is same as input — we're trying to reconstruct it (unsupervised)

        # Only train on users who rated at least one movie
        if torch.sum(target > 0).item() > 0:
            # ------------------ Forward Pass (Prediction) ------------------ #
            output = sae(input_vector)  # Pass input through AutoEncoder
            # output shape: [1, nb_movies] — reconstructed rating vector

            # ------------------ Ignore Unrated Movies ------------------ #
            target.require_grad = False          # No need to compute gradients for target

            # ------------------ ⚖️ Compute Loss (MSE) ------------------ #
            loss = criterion(output[target > 0], target[target > 0])    # MSE loss only on rated movies
            # mean_corrector normalizes RMSE by number of rated movies
            mean_corrector = nb_movies / (torch.sum(target > 0).item() + 1e-10)


            # ------------------ Backpropagation ------------------ #
            # Calculate gradients of weights w.r.t. loss
            loss.backward()

            # ------------------ Accumulate RMSE ------------------ #
            # .item() extracts scalar value from loss tensor
            train_loss += np.sqrt(loss.item() * mean_corrector)
            s += 1.  # Increment number of valid training samples
            
            # ------------------ Update Weights ------------------ #
            optimizer.step()   # Apply gradients to update weights
            optimizer.zero_grad()  # Reset gradients to zero before next user

    # Show progress after each epoch
    avg_epoch_loss = train_loss / s
    print(f'epoch: {epoch} | loss: {avg_epoch_loss:.4f}')
    losses.append(avg_epoch_loss)
    

import matplotlib.pyplot as plt

# ========================== Plot Training Loss Curve ========================== #
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training RMSE', linewidth=2.0)

# Axis labels
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('RMSE Loss', fontsize=12)

# Title and aesthetics
plt.title('Training Loss Curve', fontsize=14, weight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Add legend with better placement
plt.legend(loc='upper right', fontsize=11)

# Tight layout for cleaner spacing
plt.tight_layout()
plt.show()


# ==========================  Testing the AutoEncoder ========================== #
test_loss = 0
s = 0.

with torch.no_grad():
    for id_user in range(nb_users):
        input_vector = Variable(training_set[id_user]).unsqueeze(0)
        target = Variable(test_set[id_user]).unsqueeze(0)

        if torch.sum(target > 0).item() > 0:
            output = sae(input_vector)

            # Mask unrated movies
            output[target == 0] = 0

            # Compute RMSE only for rated movies (no mean_corrector!)
            loss = criterion(output[target > 0], target[target > 0])
            test_loss += torch.sqrt(loss).item()
            s += 1.

# Final normalized test RMSE
final_test_loss = test_loss / s
rescaled_test_loss = final_test_loss * 5  # Back to 1–5 scale

print(f'\n Test RMSE (normalized 0–1): {final_test_loss:.4f}')
print(f' Test RMSE (rescaled to 1–5): {rescaled_test_loss:.4f}')


