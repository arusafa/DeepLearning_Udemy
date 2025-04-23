# Fraud Detector

# Import the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense
from keras.models import Model

# Load the credit card transaction data
df = pd.read_csv("credit_card_transactions.csv")

# Extract the features from the data
X = df.iloc[:, :-1]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set the dimensions of the encoding layer
encoding_dim = 32

# Define the input layer
input_layer = Input(shape=(X.shape[1],))

# Define the encoding layer
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Define the decoding layer
decoded = Dense(X.shape[1], activation='sigmoid')(encoded)

# Define the autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder model
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# Use the autoencoder to reconstruct the input data
X_pred = autoencoder.predict(X_scaled)

# Calculate the reconstruction error for each sample
reconstruction_error = np.mean(np.square(X_pred - X_scaled), axis=1)

# Set a threshold for the reconstruction error
threshold = 5

# Identify the transactions that have a reconstruction error above the threshold
fraud_index = np.where(reconstruction_error > threshold)[0]

# Print the indices of the fraudulent transactions
print(fraud_index)
