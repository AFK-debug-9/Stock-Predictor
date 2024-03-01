import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

model = tf.keras.models.load_model('model.keras')

data = list(map(int,input('(last 10 stock values comma-separated) > ').strip().split(',')))
# data = [45825, 45690, 45818, 45012, 45634, 44882, 45502, 45908, 46218, 46384]

for _ in range(10):
    new_data = np.array([data[::-1][:10][::-1]])
    new_data = scaler.fit_transform(new_data.reshape(-1, 1))
    new_data = scaler.transform(new_data.reshape(-1, 1))
    new_data = new_data.reshape((1, 10, 1))

    predicted_price = model.predict(new_data)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(1, -1))

    data.append(int(predicted_price[0, 0]))

    print(f"Next Predicted Price: {predicted_price[0, 0]:.2f}")
print(data)