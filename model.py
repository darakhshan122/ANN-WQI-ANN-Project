##  Code (Step-wise Implementation)

python
# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Data
data = pd.read_csv("data.csv")
X = data.drop("WQI", axis=1)
y = data["WQI"]

# Split Data (75% Train, 15% Validation, 10% Test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4)

# Build ANN Model
model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile Model
model.compile(loss='mse', optimizer='adam')

# Train Model
model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val))

# Evaluate Model
loss = model.evaluate(X_test, y_test)
print("MSE:", loss)
