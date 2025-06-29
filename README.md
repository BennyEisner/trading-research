# Multi-Ticker LSTM Portfolio Forecasting

A functional model for forecasting next-day closing prices and building simple portfolio strategies using LSTM-based recurrent neural networks.

---

## Project Overview

- A simple LSTM model for one ticker
- Three scalable architectures for multi-ticker forecasting:
  - **Option A:** Unified Multi-Ticker LSTM
  - **Option B:** Ensemble of Single-Ticker Models
  - **Option C:** Cross-Attention Architecture

Later phases will integrate simple portfolio optimization, signal generation, and risk management

---

## Features

- LSTM sequence modeling
- Modular data loader for tickers
- multiple architectures (ensemble vs. unified vs. attention)
- portfolio strategy and risk controls

---

## Potential Future Architecture Approaches

### Option A: Unified Multi-Ticker LSTM

- Input shape: `(batch, lookback, features_per_ticker × num_tickers)`
- Single model predicts all tickers jointly

### Option B: Ensemble of Single-Ticker Models

- Independent LSTM for each ticker
- Aggregate predictions in a simple ensemble

### Option C: Cross-Attention Architecture

- Attention layers learn inter-ticker dependencies
- Most advanced, but higher compute/memory cost
