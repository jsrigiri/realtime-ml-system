# 🚀 Streaming ML Trading System (Online + Batch + XGBoost + LightGBM + Quant Metrics)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/Models-Online%20%7C%20Batch-orange)
![Boosting](https://img.shields.io/badge/Boosting-XGBoost%20%7C%20LightGBM-yellow)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Tests](https://img.shields.io/badge/Tests-Pytest-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project implements a **real-time streaming machine learning system** with:

- Online learning (tick-level updates)  
- Batch learning (session-based retraining)  
- Hybrid model architecture  
- Quantitative performance evaluation  

Supports:

- Regression → return prediction  
- Classification → directional prediction  
- Models:
  - SGD (online)
  - XGBoost
  - LightGBM  

---

## 🧠 Problem Statement

Build a system that:

- Processes streaming tick data  
- Adapts to non-stationary markets  
- Learns continuously  
- Generates tradable signals  
- Evaluates signal quality using quant metrics  

---

## 🏗 Architecture

```text
Tick Stream
   ↓
Online Feature Engineering
   ↓
Online Model (SGD)
   ↓
Signal Generation
   ↓
Execution Simulator
   ↓
Session Buffer
   ↓
Batch Model (XGBoost / LightGBM)
   ↓
Metrics + Feedback Loop
```

---

## ⚙️ Tech Stack

| Layer              | Tools |
|-------------------|------|
| Data Processing    | Pandas, NumPy |
| Online Learning    | SGDRegressor |
| Batch Models       | XGBoost, LightGBM |
| API                | FastAPI |
| Testing            | Pytest |
| Visualization      | Matplotlib |

---

## 📂 Project Structure

```text
streaming-ml-system/
├── data/
│   └── historical_ticks.csv
├── src/
│   ├── stream/
│   │   ├── data_stream.py
│   │   ├── processor.py
│   ├── features/
│   │   ├── online_features.py
│   ├── models/
│   │   ├── online_model.py
│   │   ├── batch_model.py
│   ├── execution/
│   │   ├── simulator.py
│   ├── monitoring/
│   │   ├── metrics.py
│   │   ├── plots.py
│   ├── utils/
│       ├── helpers.py
├── api/
│   ├── app.py
├── artifacts/
├── tests/
├── main.py
├── config.py
├── requirements.txt
└── README.md
```

---

## 🧠 Models

### Online Model
- SGDRegressor / SGDClassifier  
- Updates every tick  
- Handles regime shifts  

### Batch Model
- XGBoost (Regressor / Classifier)  
- LightGBM (Regressor / Classifier)  
- Trained on last N sessions  

### Hybrid Logic
- Online → fast adaptation  
- Batch → stability + structure  

---

## 📊 Quant Metrics

### Performance Metrics
- Sharpe Ratio  
- Max Drawdown  
- PnL  

### Signal Quality
- Information Coefficient (IC)  
- Rank IC  
- Signal Decay  

### Stability Metrics
- Rolling IC  
- IC Half-Life  

---

## 📈 Visualization

Generated via `monitoring/plots.py`:

- Equity curve  
- Predictions vs realized  
- Batch session PnL  
- IC curve  
- Rolling IC  

---

## ⚡ Signal Decay

Supports:

- n-tick horizon  
- k-step forward evaluation  

Measures how predictive power degrades over time.

---

## 🧪 Testing (Pytest)

Run:

```bash
pytest -v
```

### Coverage

- Data stream  
- Feature engineering  
- Models (online + batch)  
- Metrics  
- API endpoints  

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run system

```bash
python main.py
```

---

### 3. Run API

```bash
python -m uvicorn api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🔌 API Example

### Request

```json
{
  "bid": 100.1,
  "ask": 100.2,
  "mid": 100.15,
  "volume": 5
}
```

---

### Response

```json
{
  "prediction": 0.0023,
  "signal": 1
}
```

---

## 🔥 Key Highlights

- Real-time streaming ML system  
- Hybrid online + batch learning  
- XGBoost & LightGBM integration  
- Quant-style signal evaluation  
- Execution simulation  
- Strong test coverage  

---

## 🧠 Talking Points

- Built streaming ML system end-to-end  
- Designed hybrid learning architecture  
- Evaluated signals using IC + decay  
- Simulated execution and PnL  
- Balanced latency vs accuracy  

---

## 📌 Author

Machine Learning + Quant + Systems Design Portfolio Project