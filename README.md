# Hybrid Classical-Quantum Neural Network for Financial Stock Market Prediction

Our task is to build a Hybrid Quantum-Classical Recurrent Neural Network that predicts future stock index closing prices
based on historical data.  This model 
will leverage the strengths of classical Long Short-Term Memory (LSTM) networks for processing 
time-series data and integrate a trainable Variational Quantum Circuit (VQC) to enhance the 
model's feature-learning capabilities. 

The predictions will be evaluated using:
- Mean Squared Error (MSE):
- R$^2$ Score (Coefficient of Determination):
Lower MSE and higher R$^2$ indicate better performance.
## Datasets

| File | Description | 
|-----------|-----------|
| `X_train.csv` | Historical stock index data from January 1 to August 31, 2025. Includes: Date, Open, High, Low, Close, Volume. | 
| `X_test.csv` | Data for the next 10 trading days (starting September 1, 2025). Includes all columns except Close, which you must predict. | 
| `predictions.csv` | You will submit this file with your predicted Close values for each date in `X_test.csv`. | 

## Objectives

- Predict the missing Close values in `X_test.csv` using quantum machine learning.

## Installation 
The code can be run on both linux and window system. 

The working envirement and required packages can be installed with python 3.10 as following

```bash
conda create --name qml_env python=3.10
conda activate qml_env
pip install -r requirement
```

## 

## 
