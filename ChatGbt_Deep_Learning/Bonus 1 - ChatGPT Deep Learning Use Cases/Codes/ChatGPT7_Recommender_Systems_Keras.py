# Recommender Systems (Keras)

# Import the libraries
import numpy as np
import pandas as pd
from keras.layers import Embedding, Dot, Reshape, Dense
from keras.models import Sequential

# Load the movie and user data
movies = pd.read_csv('movies.csv')
users = pd.read_csv('users.csv')

# Create a mapping from movie title to movie ID
movie_ids = movies['title'].factorize()[0]

# Create a mapping from user ID to user index
user_ids = users['user_id'].factorize()[0]

# Create an embedding layer for the movie IDs
movie_embedding_size = 8
movie_embedding = Embedding(input_dim=len(movie_ids), output_dim=movie_embedding_size, input_length=1)

# Create an embedding layer for the user IDs
user_embedding_size = 8
user_embedding = Embedding(input_dim=len(user_ids), output_dim=user_embedding_size, input_length=1)

# Create a model that combines the two embeddings and predicts the rating
model = Sequential()
model.add(movie_embedding)
model.add(Reshape((movie_embedding_size,)))
model.add(user_embedding)
model.add(Reshape((user_embedding_size,)))
model.add(Dot(axes=1))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the model on the ratings data
ratings = pd.read_csv('ratings.csv')
user_indices = ratings['user_id'].apply(lambda x: user_ids[x]).values
movie_indices = ratings['movie_id'].apply(lambda x: movie_ids[x]).values
model.fit(x=[user_indices, movie_indices], y=ratings['rating'], epochs=10)

# Use the model to make recommendations for a particular user
user_index = user_ids['<user_id>']
predictions = model.predict([np.array([user_index]), np.array(range(len(movie_ids)))])
recommended_movie_ids = predictions.argsort()[-5:][::-1]
recommended_movies = movies.iloc[recommended_movie_ids]
