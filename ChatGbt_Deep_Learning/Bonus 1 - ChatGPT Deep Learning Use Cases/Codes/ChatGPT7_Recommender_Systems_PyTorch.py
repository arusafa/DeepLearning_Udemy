# Recommender Systems (PyTorch)

# Import the libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Load the movie and user data
movies = pd.read_csv('movies.csv')
users = pd.read_csv('users.csv')

# Create a mapping from movie title to movie ID
movie_ids = movies['title'].factorize()[0]

# Create a mapping from user ID to user index
user_ids = users['user_id'].factorize()[0]

# Define the model
class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, movie_embedding_size, user_embedding_size):
        super(RecommendationModel, self).__init__()
        self.movie_embedding = nn.Embedding(num_movies, movie_embedding_size)
        self.user_embedding = nn.Embedding(num_users, user_embedding_size)
        self.dense = nn.Linear(movie_embedding_size + user_embedding_size, 1)

    def forward(self, user_index, movie_index):
        movie_embedding = self.movie_embedding(movie_index)
        user_embedding = self.user_embedding(user_index)
        concatenated = torch.cat([movie_embedding, user_embedding], dim=1)
        return self.dense(concatenated).squeeze(1)

# Create the model and optimizer
model = RecommendationModel(len(user_ids), len(movie_ids), movie_embedding_size=8, user_embedding_size=8)
optimizer = optim.Adam(model.parameters())

# Train the model on the ratings data
ratings = pd.read_csv('ratings.csv')
user_indices = ratings['user_id'].apply(lambda x: user_ids[x]).values
movie_indices = ratings['movie_id'].apply(lambda x: movie_ids[x]).values
for epoch in range(10):
    for user_index, movie_index, rating in zip(user_indices, movie_indices, ratings['rating']):
        user_index = torch.tensor([user_index], dtype=torch.long)
        movie_index = torch.tensor([movie_index], dtype=torch.long)
        rating = torch.tensor([rating], dtype=torch.float)
        prediction = model(user_index, movie_index)
        loss = nn.MSELoss()(prediction, rating)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Use the model to make recommendations for a particular user
user_index = user_ids['<user_id>']
predictions = []
for movie_index in range(len(movie_ids)):
    movie_index = torch.tensor([movie_index], dtype=torch.long)
    predictions.append(model)