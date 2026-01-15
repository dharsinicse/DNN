import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
data = pd.read_csv("student_data.csv")
print(data)

# Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_len=3):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)

# Build LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(X.shape[2]))
model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(X, y, epochs=200, verbose=1)