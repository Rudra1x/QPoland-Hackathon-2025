# Hybrid Classical-Quantum Neural Network for Financial Stock Market Prediction
## Team - Beerantum [ Van Binh VU & Rudraksh Sharma ]

Our task is to build a Hybrid Quantum-Classical Recurrent Neural Network that predicts future stock index closing prices
based on historical data.  This model 
will leverage the strengths of classical Long Short-Term Memory (LSTM) networks for processing 
time-series data and integrate a trainable Variational Quantum Circuit (VQC) to enhance the 
model's feature-learning capabilities. 

The predictions will be evaluated using:
- Mean Squared Error (MSE):
- R^2 Score (Coefficient of Determination):
Lower MSE and higher R^2 indicate better performance.
## Datasets

| File | Description | 
|-----------|-----------|
| `X_train.csv` | Historical stock index data from January 1 to August 31, 2025. Includes: Date, Open, High, Low, Close, Volume. | 
| `X_test.csv` | Data for the next 10 trading days (starting September 1, 2025). Includes all columns except Close, which you must predict. | 
| `predictions.csv` | You will submit this file with your predicted Close values for each date in `X_test.csv`. | 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Framework: Qiskit](https://img.shields.io/badge/Framework-Qiskit-blue.svg)](https://qiskit.org/)

## Objectives

This repository contains a novel hybrid Quantum-Classical model for time-series forecasting, specifically designed to predict stock prices (the missing Close values in `X_test.csv`) with higher accuracy by using a quantum circuit to correct classical errors.

  
---

## 💡 The Core Idea: Quantum Residual Correction

Classical models like LSTMs are powerful for sequence prediction, but they often struggle with the complex, non-linear, and near-chaotic dynamics of financial markets. Their predictions inevitably contain errors (residuals).

Our hypothesis is that these residuals, while seemingly random, contain subtle, complex patterns that a classical model cannot capture. We use a **Variational Quantum Circuit (VQC)** to learn these patterns.

Our architecture is simple and powerful:

**Final Prediction = Classical Prediction (LSTM) + Quantum Solution (VQC)**

This allows the classical model to do the "heavy lifting" while the quantum model acts as a specialized, high-performance corrector, giving us the best of both worlds.


---
Structure
----Hybrid Model
---------Model1
---------Model2
----Requirement.txt
----Final Project Report

## 🔬 The Models

This repository contains two notebooks, each detailing a version of our QRC architecture:

1.  **`model1/main.ipynb` - [Baseline Hybrid Model]**: Implements the QRC architecture using a custom-built Variational Quantum Circuit (VQC) from basic Qiskit gates (`ry`, `cx`).
2.  **`model2/main.ipynb` - [Enhanced Hybrid Model]**: Implements a more advanced and structured version using Qiskit's standard `ZZFeatureMap` for data encoding and the `TwoLocal` ansatz as the trainable circuit.

Both models follow the same robust 4-phase pipeline.

---

## 🛠 Project Pipeline: A 4-Phase Approach

Our workflow is broken down into four distinct, reproducible phases:

### Phase 1: Advanced Data Engineering
We don't just feed in raw prices. We build a rich, contextual dataset to give our models the best possible chance to find patterns.
* **Advanced Feature Engineering**: We create several technical indicators, including **Relative Strength Index (RSI)**, **Bollinger Bands**, Moving Averages, and Lagged Features.
* **Robust Scaling**: We use `RobustScaler` from scikit-learn. Unlike standard scalers, it is resilient to the extreme outliers often found in financial data, leading to more stable model training.
* **Time-Series Windowing**: We structure the data into sequences of 20 timesteps (days) to predict the 'Close' price of the 21st day.

### Phase 2: The Quantum-Classical Architecture
This is the heart of our project. Our `QuantumResidualModel` (built in PyTorch) consists of two components that are trained *simultaneously*:

1.  **The Classical Backbone**: A 2-layer, **Bidirectional LSTM** with Dropout. It processes the 20-day data window and generates a primary (classical) prediction.
2.  **The Quantum Corrector**: A 4-qubit **Variational Quantum Circuit (VQC)**.
    * The final hidden state of the LSTM is passed through a small linear layer (`q_input_scaler`) to match the VQC's feature dimension.
    * The VQC (implemented in `model1/main` with `ParameterVector` and in `model2/main` with `ZZFeatureMap` + `TwoLocal`) learns to predict the residual (the error) of the classical model's prediction.
    * We use **Qiskit's `EstimatorQNN`** and **`TorchConnector`** to make the quantum circuit a seamless, differentiable layer within our PyTorch model.

### Phase 3: State-of-the-Art Training
A great model needs a great training strategy.
* **Optimizer**: We use **`AdamW`**, an improved version of Adam that provides better regularization and weight decay.
* **Scheduler**: We implement **`CosineAnnealingWarmRestarts`**, an advanced learning rate scheduler that cyclically adjusts the learning rate. This helps the model converge faster and escape local minima, leading to a more optimal solution.
* **Stability**: We use **Gradient Clipping** (`clip_grad_norm_`) to prevent the exploding gradient problem, which is common in LSTMs and recurrent models.
* The best model weights are saved to `qiskit_residual_model.pth`.

### Phase 4: Prediction & Evaluation
We load the trained model and perform a robust iterative prediction loop on the test data (`X_test.csv`) to generate the final `predictions.csv` file.

---

## 📊 Results & Performance

Our hybrid model demonstrates strong predictive capability, achieving an **R-squared (R²) score of 0.705** on the validation set. The visualizations generated by the notebooks confirm that our model's predictions (red) effectively track the actual stock prices (blue).

### Actual vs. Predicted Performance
## Model 1 
<img width="1161" height="620" alt="image" src="https://github.com/user-attachments/assets/cd0977b7-e4c6-4df9-9710-286e149dff2c" />
<img width="1067" height="370" alt="image" src="https://github.com/user-attachments/assets/f75e6840-502d-42ed-90e1-f8d15fe45be3" />

## Model 2 
<img width="1161" height="622" alt="image" src="https://github.com/user-attachments/assets/42a053eb-ba02-4a4d-b8b6-b960b803c079" />
<img width="1212" height="417" alt="image" src="https://github.com/user-attachments/assets/586b8963-7add-48ff-84d6-e7c063888ea1" />

---
## Model Architecture 
## Model 1 <img width="1193" height="1590" alt="model1" src="https://github.com/user-attachments/assets/64974f3c-e424-4f25-a778-7a6cd69fbb4a" />
## Model 2 <img width="1248" height="1590" alt="model2" src="https://github.com/user-attachments/assets/906e43e6-3ae7-43cd-a145-9774c62c3599" />


## 🚀 How to Run

The code can be run on both linux and window system. 

The working envirement and required packages can be installed with python 3.10 as following

```bash
conda create --name qml_env python=3.10
conda activate qml_env
pip install -r requirement
```
MIT License

Copyright (c) 2025 Beerantum
## 

## 
