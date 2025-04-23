#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:46:39 2025
@author: safaaru
"""

# ------------------- Boltzmann Machines for Movie Recommendation -------------------

# Importing required libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# ------------------- Step 1: Load the Dataset -------------------

# Reading MovieLens 1M metadata files
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Reading the training and test set from 100k subset
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# ------------------- Step 2: Process the Data -------------------

# Find the total number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Convert user-item ratings to a matrix format (users as rows, movies as columns)
def convert(data):
    new_data = []
    for id_user in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_user]
        id_ratings = data[:, 2][data[:, 0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert the dataset to PyTorch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# ------------------- Step 3: Binary Conversion of Ratings -------------------

# Convert ratings to binary values: 1 (Liked), 0 (Not liked), -1 (Unrated)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# ------------------- Step 4: RBM Model Class -------------------

class RBM():
    def __init__(self, nv, nh):
        # nv: number of visible nodes (movies)
        # nh: number of hidden nodes (features)
        self.W = torch.randn(nh, nv)        # Weight matrix [nh x nv]
        self.a = torch.randn(1, nh)         # Hidden layer bias [1 x nh]
        self.b = torch.randn(1, nv)         # Visible layer bias [1 x nv]

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())        # Linear activation to hidden
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)            # Linear activation to visible
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        # Weight update using Contrastive Divergence
        self.W += torch.mm(ph0.t(), v0) - torch.mm(phk.t(), vk)
        self.b += torch.sum((v0 - vk), dim=0)
        self.a += torch.sum((ph0 - phk), dim=0)

# ------------------- Step 5: Initialize the RBM -------------------

nv = len(training_set[0])  # Number of visible nodes (movies)
nh = 100                   # Number of hidden nodes (features)
batch_size = 100           # Batch size

rbm = RBM(nv, nh)

# ------------------- Step 6: Train the RBM -------------------

nb_epoch = 10

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.  # Sample counter

    for id_user in range(0, nb_users - batch_size, batch_size):
        v0 = training_set[id_user: id_user + batch_size]  # Original input
        vk = training_set[id_user: id_user + batch_size]  # Reconstructed input
        ph0, _ = rbm.sample_h(v0)

        for k in range(10):  # Contrastive Divergence k-steps
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]  # Preserve unrated positions

        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)

        # Compute average loss (only on rated movies)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.

    print(f'Epoch: {epoch} | Loss: {train_loss.item() / s:.4f}')


# ------------------- Step 6: Testing the RBM -------------------

test_loss = 0
s = 0.

for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]  # Training data (input)
    vt = test_set[id_user:id_user+1]     # Ground truth (labels)

    if len(vt[vt >= 0]) > 0:  # Evaluate only if test set has ratings
        _, h = rbm.sample_h(v)
        _, v_reconstructed = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v_reconstructed[vt >= 0]))
        s += 1.

print(f'\nTest Loss: {test_loss.item() / s:.4f}')


# ------------------- Step 7: Recommend Top-N movies in the RBM -------------------

def recommend_movies(user_id, rbm, top_n=10):
    """
    Recommend Top-N movies for a specific user using the trained RBM model.

    Args:
        user_id (int): ID of the user (1-indexed as per MovieLens dataset).
        rbm (RBM): Trained Restricted Boltzmann Machine model.
        top_n (int): Number of movie recommendations to return.

    Returns:
        None: Prints the movie titles recommended for the user.
    """

    # Step 1: Get the input vector (user's rating history) from the training set
    # Note: subtract 1 from user_id because indexing in training_set starts from 0
    user_vector = training_set[user_id - 1:user_id]  # Shape: [1, num_movies]

    # Step 2: Feed the user input through the RBM to get hidden features
    _, h = rbm.sample_h(user_vector)  # h shape: [1, num_hidden]

    # Step 3: Reconstruct visible units (i.e., predicted ratings) from hidden layer
    _, predicted_ratings = rbm.sample_v(h)  # Shape: [1, num_movies]

    # Step 4: Mask the movies the user has already rated
    # This ensures we only recommend movies the user hasn't seen
    rated_movies = training_set[user_id - 1] > 0  # Boolean mask of rated movies
    predicted_ratings[0][rated_movies] = -1      # Set already-rated movies to -1

    # Step 5: Get indices of the top N highest predicted ratings
    # These are the most strongly recommended unseen movies
    recommended_movies = torch.topk(predicted_ratings[0], top_n).indices.numpy()

    # Step 6: Print the recommended movie titles using the movie IDs
    print(f'\nTop {top_n} Recommended Movies for User {user_id}:')
    for idx in recommended_movies:
        print(f"- {movies.iloc[idx, 1]}")  # movies.iloc[index, 1] gives the movie title

recommend_movies(user_id=25, rbm=rbm, top_n=5)



# ------------------- Step 7: Recommend Top-N movies group by Genre in the RBM -------------------

def recommend_movies_by_genre(user_id, rbm, top_n=10):
    """
    Recommend Top-N movies for a user using RBM, grouped by genre.

    Args:
        user_id (int): User ID from 1 to nb_users.
        rbm (RBM): Trained RBM model.
        top_n (int): Number of total recommendations.

    Returns:
        None: Prints sorted recommendations by genre.
    """
    # Get user's rating vector (1 row only)
    user_vector = training_set[user_id - 1:user_id]

    # Sample hidden values from user's rating
    _, h = rbm.sample_h(user_vector)

    # Reconstruct the visible layer (i.e., predicted movie preferences)
    _, predicted_ratings = rbm.sample_v(h)

    # Filter out already rated movies
    rated_movies = training_set[user_id - 1] > 0
    predicted_ratings[0][rated_movies] = -1

    # Get top-N movie indices
    top_indices = torch.topk(predicted_ratings[0], top_n).indices.numpy()

    # Prepare a genre-sorted dictionary
    genre_map = {}

    print(f'\nTop {top_n} Movie Recommendations for User {user_id}, Sorted by Genre:\n')

    for idx in top_indices:
        movie_title = movies.iloc[idx, 1]
        genres = movies.iloc[idx, 2].split('|')  # Can be multiple genres like "Action|Sci-Fi"

        for genre in genres:
            if genre not in genre_map:
                genre_map[genre] = []
            genre_map[genre].append(movie_title)

    # Display recommendations by genre
    for genre in sorted(genre_map):
        print(f"\n🎬 {genre} ({len(genre_map[genre])} movie{'s' if len(genre_map[genre]) > 1 else ''})")
        for title in genre_map[genre]:
            print(f"  - {title}")

recommend_movies_by_genre(user_id=42, rbm=rbm, top_n=10)






