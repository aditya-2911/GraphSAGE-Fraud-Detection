# Inductive Graph Neural Network for Crypto Fraud Detection

This repository contains an implementation of GraphSAGE (Inductive Representation Learning on Large Graphs) applied to the Elliptic Bitcoin dataset for fraud detection.

## Overview
- **Dataset**: Elliptic Bitcoin Dataset (nodes = transactions, edges = payment flows).
- **Task**: Node classification to detect illicit transactions (Fraud Detection).
- **Model**: GraphSAGE vs Baseline MLP.
- **Evaluation**: Inductive learning (trained on past transactions, evaluated on temporally unseen future transactions).

## Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Model
```bash
python src/main.py
```
