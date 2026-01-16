# 📈 HCMC Apartment Price Prediction (Bi-LSTM)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat-square&logo=Matplotlib&logoColor=black)

Welcome to my project! This repository contains a Deep Learning model designed to predict real estate prices in **Ho Chi Minh City (HCMC)** based on historical data.

The core of this project is a **Bidirectional LSTM (Bi-LSTM)** model implemented in PyTorch.

--- --- 
## 🚀 Key Feature: Bi-LSTM
I have upgraded the standard LSTM model to a **Bi-LSTM (Bidirectional Long Short-Term Memory)** architecture.

### Why Bi-LSTM?
- **Standard LSTM:** Only learns from the past to the future (Forward).
- **Bi-LSTM:** Learns in two directions at the same time:
- **Forward** (Past $\to$ Future)
- **Backward** (Future $\to$ Past)

This allows the model to understand the context of the price trends much better than a basic LSTM.

--- --- 
## 📂 Repository Structure

### 1. Source Code 💻
- `main.py:` The main file. It runs the training loop, evaluates the model, and plots the results.
- `lstm_model.py:` Contains the BiLSTMModel class (The neural network architecture).
- `data_loader.py:` Handles data preprocessing, normalization (MinMaxScaling), and creating time-series sequences.
- `generate_data.py:` Helper script to process or generate dataset samples.

### 2. Datasets 📊
- `Dataset_BDS_HCM_Merged.csv:` The main dataset containing real estate prices in Ho Chi Minh City.
- `Dataset_BDS_HCM_Diff.csv:` Processed data (difference transformation) to make the time series stationary.
- `Dataset_GCC_HCM_2015_2025.csv:` Additional economic/construction data for the period 2015-2025.

--- --- 
## 🛠️ Model Architecture
The model in `lstm_model.py` is built using PyTorch:
```python
def step2_build_model(self):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(128, return_sequences=True), #Direction
                                     input_shape=(self.look_back, self.n_features)))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(64, return_sequences=False))) #Parameter
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1))  # Output layer
        self.model.compile(optimizer='adam', loss='mean_squared_error')
```

--- --- 
## ⚙️ How to Run
### 1. Install dependencies: You need Python and the following libraries:
- torch (PyTorch)
- pandas
- scikit-learn
- matplotlib
### 2. Run the training:
```bash
python main.py
```
### 3. Result: The script will train the model and display a chart comparing Predicted Prices vs. Actual Prices.
<img width="1064" height="604" alt="image" src="https://github.com/user-attachments/assets/345a130c-2c15-4e4a-8f12-fe35fc385e38" />
<img width="835" height="608" alt="image" src="https://github.com/user-attachments/assets/ff4b6b74-2902-4a7d-bd2f-b124d0a2facc" />
<img width="847" height="608" alt="image" src="https://github.com/user-attachments/assets/ee35ec54-bd2f-42ee-ab4b-c15d48e13854" />

--- --- 
## 👨‍💻 Author
- Project: HCMC Apartment Price Prediction LTSM
- Teammate: Ho Dang Thai Duy, Le Minh Hoang, Tran Duc Anh, Tran Xuan An, Nong Hoang Anh

--- ---
## 🌟 Thanks for checking out us project!
