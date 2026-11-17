# Climate-Resilient Food System Predictive Modeling — DSC RASIO 10.0

![ci](https://github.com/henryferdianto/dsc-rasio-10-food-system/actions/workflows/ci.yml/badge.svg)

> **Predicting Southeast Asian food system resilience under climate variability**

## Overview

Predictive machine learning models that assess Southeast Asian food system resilience under climate variability, using environmental and socioeconomic features and gradient-boosted decision trees.

## Approach

1. **Data integration** of agro-climatic and socioeconomic indicators.
2. **Multi-regional feature engineering** (climate anomalies, yield trends).
3. **Modeling** with gradient-boosted trees and regularized regression.
4. **Evaluation** with RMSE/MAE and cross-region validation.

## Repository Structure

```
dsc-rasio-10-food-system/
├── src/features.py
├── src/train.py
├── requirements.txt
└── README.md
```

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `LightGBM`

## Getting Started

```bash
git clone https://github.com/henryferdianto/dsc-rasio-10-food-system.git
cd dsc-rasio-10-food-system
pip install -r requirements.txt
```

```bash
python src/train.py --input data/raw/food_system.csv --target resilience_index
```


<!-- completed -->

## Quickstart with Sample Data

The repository ships a synthetic data generator so the full pipeline can be run
without any private dataset:

```bash
make sample && python -m src.train
python -m src.run_demo
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```

## Project Status

This repository contains the project documentation and a runnable reference
implementation. A synthetic data generator is included so every pipeline stage
can be executed end-to-end; replace the sample data with your own datasets in
`data/raw/` to reproduce real results.

## License

MIT License — see [LICENSE](LICENSE).


---

**Author:** Rifky Henry Ferdianto — Data Science undergraduate at ITERA  
GitHub: [@henryferdianto](https://github.com/henryferdianto) · Portfolio: [rifky-henry-ferdianto](https://github.com/henryferdianto/rifky-henry-ferdianto)
