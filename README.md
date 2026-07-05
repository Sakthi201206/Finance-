# Volatility-Regimes & Alpha Trend Predictor

A lightweight, production-ready machine learning pipeline developed to extract predictive trend signals and identify short-term asset outperformance (Alpha) in the Indian equity markets. 

This infrastructure bypasses traditional wrapper constraints by computing custom financial indicators manually, evaluating feature variance and trend metrics to categorize market states.

## 🛠️ Pipeline Architecture

1. **Data Ingestion**: Programmatic retrieval of historical, non-stationary daily market data via the Yahoo Finance API (`yfinance`).
2. **Feature Engineering & Processing**:
   * **Stationarity Normalization**: Conversion of raw asset pricing trajectories into scaled daily rate-of-change sequences (Percentage Returns).
   * **Variance Extraction**: Computation of rolling standard deviations over short-term (5-day) and medium-term (20-day) lookback windows to capture immediate regime volatility shifts.
   * **Signal Smoothing**: Low-pass filtering utilizing Exponential Moving Average (EMA) crossovers (`EMA_9 - EMA_21`) to isolate clean structural trends from market noise.
   * **Bounded Momentum Scaling**: Algorithmic implementation of the Relative Strength Index (RSI) bounded strictly between `0.0` and `100.0`.
3. **Chronological Validation**: Split architecture maintains sequential timeline integrity (80/20 train-test allocation), completely eliminating look-ahead bias and structural data leakage.
4. **Classification Infrastructure**: Training an optimized eXtreme Gradient Boosting (`XGBClassifier`) algorithm to map feature matrices to forward-looking fixed-horizon outcomes.

## 📊 Objective Functions & Target Generation

The model optimizes for high precision over a 5-day predictive horizon. The predictive target evaluates a continuous future window profile and translates it into a binary classification setup:

$$\text{Target} = \begin{cases} 1, & \text{if Forward 5-Day Return} > 1.5\% \\ 0, & \text{otherwise} \end{cases}$$

## 🚀 Execution Instructions

### Installation
```bash
pip install yfinance pandas scikit-learn xgboost numpy
```

### Run Model
```bash
python main.py
```

## 🛠️ Technical Competencies Demonstrated
* Time-Series Feature Engineering & Preprocessing
* Advanced Signal Smoothing & Volatility Isolation
* Strict Chronological Model Validation Protocols
* Gradient Boosted Tree Architecture Optimization (`XGBoost`)
