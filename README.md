Stock Price Forecasting Model
This repository contains a stock price forecasting model using time series data from NTT's stock prices. The model is built using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN), to predict future stock prices based on historical data.

Table of Contents
Project Overview
Data
Installation
Usage
Model Architecture
Model Training
Saving the Model
Loading and Using the Model
License
Project Overview
The purpose of this project is to develop a time series forecasting model that can predict future stock prices using NTT stock price data. The model is built using LSTM to capture the sequential nature of stock price data. The project involves:

Exploratory Data Analysis (EDA) to understand the dataset.
Data preprocessing and feature engineering.
Model building and training using LSTM.
Model evaluation and improvement.
Saving and loading the model for future use.
Data
The dataset contains daily stock price data for NTT, including the following features:

Close Price: The adjusted closing price for each day.
Open Price: The price at which the stock opened on the day.
High Price: The highest price the stock reached on the day.
Low Price: The lowest price the stock reached on the day.
Volume: The number of shares traded during the day.
Change %: The percentage change in price from the previous day.
Installation
Prerequisites
Python 3.x

Required Libraries: Install the dependencies using the following command:

bash
Copy code
pip install -r requirements.txt
Dependencies
pandas
numpy
keras
scikit-learn
matplotlib
Usage
Clone the Repository:

bash
Copy code
git clone https://github.com/your-username/stock-price-forecasting.git
cd stock-price-forecasting
Run the Jupyter Notebook: Open the provided Jupyter notebook file Stock_Price_Forecasting_Model.ipynb to run the project step-by-step.

Model Training: Train the model by running the LSTM model training steps within the notebook.

Save the Model: Once the model is trained, you can save it using:

python
Copy code
model.save('stock_price_model.h5')
Load the Model: After saving, the model can be loaded using the following command:

python
Copy code
from keras.models import load_model
model = load_model('stock_price_model.h5')
Make Predictions: Use the loaded model to make stock price predictions:

python
Copy code
predictions = model.predict(X_test)
Model Architecture
The LSTM model used in this project consists of the following layers:

LSTM layer to capture sequential dependencies in the data.
Dense layer for the final output.
Example of Model Architecture:
python
Copy code
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
Model Training
Training the LSTM model on the stock price data:

Epochs: 10
Batch Size: 32
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam
python
Copy code
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
Saving the Model
After training, save the model for future use:

python
Copy code
model.save('stock_price_model.h5')
Loading and Using the Model
To load the saved model and make predictions, use the following:

python
Copy code
from keras.models import load_model

# Load the model
model = load_model('stock_price_model.h5')

# Make predictions
predictions = model.predict(X_test)
License
This project is licensed under the MIT License. See the LICENSE file for details.
