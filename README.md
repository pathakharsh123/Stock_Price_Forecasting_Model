# Stock Price Forecasting Model

This repository contains a stock price forecasting model using time series data from NTT's stock prices. The model is built using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN), to predict future stock prices based on historical data.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Saving the Model](#saving-the-model)
- [Loading and Using the Model](#loading-and-using-the-model)
- [License](#license)

## Project Overview

The purpose of this project is to develop a time series forecasting model that can predict future stock prices using NTT stock price data. The model is built using LSTM to capture the sequential nature of stock price data. The project involves:

1. Exploratory Data Analysis (EDA) to understand the dataset.
2. Data preprocessing and feature engineering.
3. Model building and training using LSTM.
4. Model evaluation and improvement.
5. Saving and loading the model for future use.

## Data

The dataset contains daily stock price data for NTT, including the following features:
- **Close Price**: The adjusted closing price for each day.
- **Open Price**: The price at which the stock opened on the day.
- **High Price**: The highest price the stock reached on the day.
- **Low Price**: The lowest price the stock reached on the day.
- **Volume**: The number of shares traded during the day.
- **Change %**: The percentage change in price from the previous day.

## Installation

### Prerequisites
- Python 3.x
- Required Libraries: Install the dependencies using the following command:
  
  ```bash
  pip install -r requirements.txt
  ## Dependencies

To install the required libraries, make sure your `requirements.txt` file contains the following:

```txt
pandas
numpy
keras
scikit-learn
matplotlib
