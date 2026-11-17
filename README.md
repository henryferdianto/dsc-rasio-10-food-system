# Climate-Resilient Food System Predictive Modeling — DSC RASIO 10.0

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

## Project Status

This repository contains the project documentation and a runnable implementation
skeleton. Scripts are structured so that each pipeline stage can be executed and
extended independently. Update the configuration and data paths to run against
your own datasets.

## License

MIT License — see [LICENSE](LICENSE).


---

**Author:** Rifky Henry Ferdianto — Data Science undergraduate at ITERA  
GitHub: [@henryferdianto](https://github.com/henryferdianto) · Portfolio: [rifky-henry-ferdianto](https://github.com/henryferdianto/rifky-henry-ferdianto)
