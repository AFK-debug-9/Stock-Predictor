import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

stock_data = pd.read_csv("data.csv")

# Use the last three closing prices as features
data = stock_data['Price'].values
features = []
labels = []

for i in range(len(data) - 11):
    features.append(data[i:i + 10])
    labels.append(data[i + 11])

features = np.array(features)
labels = np.array(labels)

scaler = MinMaxScaler()
features = scaler.fit_transform(features.reshape(-1, 1))
labels = scaler.transform(labels.reshape(-1, 1))

features = features.reshape((-1, 10))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, activation='relu', input_shape=(10, 1), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001), loss='mean_squared_error')

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model.fit(X_train, y_train, epochs=150, batch_size=64,
          validation_data=(X_test, y_test))

loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

model.save('model.keras')