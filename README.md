# 🚀 Streaming ML Trading System (Online + Batch + XGBoost + LightGBM)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streaming](https://img.shields.io/badge/System-Streaming%20ML-orange)
![Models](https://img.shields.io/badge/Models-Online%20%7C%20Batch-green)
![Boosting](https://img.shields.io/badge/Boosting-XGBoost%20%7C%20LightGBM-yellow)
![Tests](https://img.shields.io/badge/Tests-Pytest-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 📌 Overview

This project builds a **real-time streaming machine learning trading system** combining:

- Online learning (tick-by-tick updates)
- Batch learning (session-level retraining)
- Hybrid feedback loop between models

Supports regression, classification, XGBoost, LightGBM, GPU fallback, and full pytest coverage.

---

## 🏗 Architecture

Tick Stream → Online Features → Online Model → Trading Engine → Execution  
→ Session Buffer → Batch Model → Metrics → Feedback Loop

---

## ⚙️ Tech Stack

- Python, Pandas, NumPy  
- Scikit-learn, XGBoost, LightGBM  
- FastAPI (optional)  
- Pytest  

---

## 📂 Project Structure

```
streaming-ml-system/
├── src/
├── tests/
├── main.py
├── config.py
└── README.md
```

---

## 🧠 Models

- Online: SGDRegressor  
- Batch:
  - Linear / Logistic
  - XGBoost
  - LightGBM  

---

## ▶️ Run

```
pip install -r requirements.txt
python main.py
pytest -v
```

---

## 📈 Features

- Real-time ML  
- Hybrid online + batch learning  
- k-step ahead prediction  
- Trading simulation  
- GPU support  

---

## 📌 Author

Machine Learning + Quant Portfolio Project
